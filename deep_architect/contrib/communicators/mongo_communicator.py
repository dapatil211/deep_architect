from datetime import datetime, timedelta
import time
import threading
from multiprocessing import Process
import logging

from pymongo import MongoClient, ReturnDocument

logger = logging.getLogger(__name__)

STARTTIME_KEY = 'startTime'
ENDTIME_KEY = 'endTime'
REFRESH_KEY = 'last_refreshed'
TIME_WORKED_KEY = 'totalTime'
DATA_KEY = 'data'


class MongoCommunicator(object):
    """A communicator for distributed running of DeepArchitect that is based on
    a MongoDB database. The communicator is used to implement a master/worker
    paradigm where the master pushes jobs to a work queue which are consumed by
    the workers. The workers then send back results to the master. The
    communicator is structured so that it can only process one job at a time per
    subscription.
    """

    def __init__(self,
                 host='localhost',
                 port=27017,
                 refresh_period=30,
                 data_refresher=False):
        """Constructs the MongoDB based communicator.
        Arguments:
            - host: The host where the MongoDB server is running
            - port: The port where the MongoDB server is running
            - refresh_period: The period of time in seconds between refreshing
                data stored in the intermediate tables. If there is data in the
                intermediate tables that has not been refreshed in this period,
                it is moved back to the original table.
        """
        self.host = host
        self.port = port
        self._client = MongoClient(host, port)
        self._db = self._client['deep-architect']
        self._subscribed = {}
        self._processing = {}
        self._refresh_period = refresh_period
        self._refresh_thread = None
        if data_refresher:
            self.refresh_process = Process(target=self._move_stale_data)
            self.refresh_process.daemon = True
            self.refresh_process.start()

    def __del__(self):
        if self._refresh_thread is not None:
            self._refresh_thread.terminate()

    def _move_stale_data(self):
        time.sleep(2)
        client = MongoClient(self.host, self.port, connect=False)
        db = client['deep-architect']
        while True:
            logger.debug('Resetting stale data in DB')
            for collection in db.list_collection_names():
                collection = db[collection]
                cursor = collection.find({
                    '$and': [{
                        REFRESH_KEY: {
                            '$lt':
                            datetime.now() -
                            timedelta(seconds=self._refresh_period + 10)
                        }
                    }, {
                        STARTTIME_KEY: {
                            '$ne': None
                        }
                    }, {
                        ENDTIME_KEY: {
                            '$eq': None
                        }
                    }]
                })
                for doc in cursor:
                    logger.info('Resetting %s in %s', doc['_id'],
                                collection.name)
                    if TIME_WORKED_KEY not in doc:
                        doc[TIME_WORKED_KEY] = 0.0
                    doc[TIME_WORKED_KEY] += (
                        doc[REFRESH_KEY] - doc[STARTTIME_KEY]).total_seconds()
                    doc[STARTTIME_KEY] = None
                    doc[ENDTIME_KEY] = None
                    doc[REFRESH_KEY] = None
                    collection.update_one(
                        {
                            '_id':
                            doc['_id'],
                            '$and': [{
                                REFRESH_KEY: {
                                    '$lt':
                                    datetime.now() -
                                    timedelta(seconds=self._refresh_period + 10)
                                }
                            }, {
                                STARTTIME_KEY: {
                                    '$ne': None
                                }
                            }, {
                                ENDTIME_KEY: {
                                    '$eq': None
                                }
                            }]
                        }, {'$set': doc})
            time.sleep(self._refresh_period)

    def publish(self, topic, data):
        """Publishes data to some topic. Blocking call. Data is put under 'data'
            key.
        Arguments:
            - topic: The topic to publish the data to
            - data: bson compatible object with the data to publish to topic
        """
        collection = self._db[topic]
        collection.insert_one({
            DATA_KEY: data,
            STARTTIME_KEY: None,
            ENDTIME_KEY: None
        })

    def subscribe(self, subscription, callback):
        """Subscribes to some topic. Non-blocking call. If store_intermediate is
        True, then after each message is consumed and finished processing,
        finish_processing must be called with the original message.
        Arguments:
            - subscription: The name of the topic to subscribe to.
            - callback: Function that is called with dictionary where the object
                data is under the 'data' key.
            - store_intermediate: This parameter controls whether intermediate
                job configs are stored while they are being processed
        """
        if subscription in self._subscribed:
            raise RuntimeError('Already subscribed to this subscription')
        logger.info('Subscribed to %s', subscription)
        self._subscribed[subscription] = True
        thread = threading.Thread(target=self._subscribe,
                                  args=(subscription, callback))
        thread.start()

    def _subscribe(self, subscription, callback):
        collection = self._db[subscription]
        self._processing[subscription] = False

        while self._subscribed[subscription]:
            # The current job is still being processed
            if self._processing[subscription]:
                time.sleep(10)
                continue

            data = collection.find_one_and_update(
                {STARTTIME_KEY: {
                    '$eq': None
                }}, {'$currentDate': {
                    REFRESH_KEY: True,
                    STARTTIME_KEY: True
                }},
                return_document=ReturnDocument.AFTER)
            # Nothing currently in the subscription queue
            if data is None:
                time.sleep(10)
            else:
                self._processing[subscription] = True

                refresh_process = Process(target=refresh_data,
                                          args=(data, collection.name,
                                                self._refresh_period, self.host,
                                                self.port))
                refresh_process.daemon = True
                refresh_process.start()

                callback(data)

    def finish_processing(self, subscription, data, success=True):
        """Removes the message from the intermediate processing storage. Must be
        called for every message received if store_intermediate is True.
        Arguments:
            - subscription: The name of the topic to subscribe to.
            - callback: Function that is called with the object representing
                the data that was consumed.
            - success: whether the processing of the message was successful. If
                not, the data is put back into the original queue
        """
        logger.info('Finish processing %s', str(data))
        collection = self._db[subscription]
        if success:
            data = collection.find_one_and_update(
                {'_id': data['_id']}, {
                    '$currentDate': {
                        ENDTIME_KEY: True
                    },
                    '$inc': {
                        TIME_WORKED_KEY: (data[REFRESH_KEY] -
                                          data[STARTTIME_KEY]).total_seconds()
                    }
                },
                return_document=ReturnDocument.AFTER)
            logger.info('Successfully finished %s from %s', str(data['_id']),
                        subscription)
        else:
            collection.find_one_and_update({'_id': data['_id']}, {
                '$set': {
                    STARTTIME_KEY: None,
                    ENDTIME_KEY: None,
                    REFRESH_KEY: None
                }
            })
            logger.info('Unsuccessful processing %s, reinserting into %s',
                        str(data['_id']), subscription)
        self._processing[subscription] = False

    def unsubscribe(self, subscription):
        """Stops communicator from listening to subscription:
            - subscription: The name of the topic to unsubscribe from.
        """
        logger.info('Unsubscribed from %s', subscription)
        self._subscribed[subscription] = False

    def check_data_exists(self, subscription, key, value):
        """Checks if the subscription contains data with a given key and value
            - subscription: The name of the subscription in which data may be
                located
            - key: string with data key
            - value: MongoDB compatible data type with value corresponding to
                key
        """
        if not isinstance(key, str):
            raise ValueError('Lookup key must be string')
        results = self._db[subscription].find_one({'data.' + key: value})
        return results is not None

    def update(self, subscription, data, key, value):
        """ Finds entry with the same id as data and updates its key with value
            - subscription: The name of the subscription in which data may be
                located
            - data: dictionary with '_id' key containing an ObjectId
                corresponding to a MongoDB object
            - key: string with data key
            - value: MongoDB compatible data type with value corresponding to
                key
        """
        if not isinstance(key, str):
            raise ValueError('Lookup key must be string')
        self._db[subscription].find_one_and_update(
            {'_id': data['_id']}, {'$set': {
                'data.' + key: value
            }})

    def get_value(self, subscription, key, value):
        """Returns data with a given key and value in a subscription. Returns
            None if data doesn't exist.
            - subscription: The name of the subscription in which data may be
                located
            - key: string with data key
            - value: MongoDB compatible data type with value corresponding to
                key
        """
        if not isinstance(key, str):
            raise ValueError('Lookup key must be string')
        results = self._db[subscription].find_one({'data.' + key: value})
        return results


def refresh_data(data, collection_name, refresh_period, host, port):
    client = MongoClient(host, port)
    db = client['deep-architect']
    collection = db[collection_name]
    while collection.find_one_and_update(
        {
            '$and': [{
                '_id': data['_id']
            }, {
                STARTTIME_KEY: {
                    '$ne': None
                }
            }, {
                ENDTIME_KEY: {
                    '$eq': None
                }
            }]
        }, {'$currentDate': {
            REFRESH_KEY: True
        }},
            return_document=ReturnDocument.AFTER):
        logger.debug('Refreshed data for id %s', str(data['_id']))
        time.sleep(refresh_period)
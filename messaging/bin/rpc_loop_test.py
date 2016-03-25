#!/usr/bin/env python

import sys
import socket
import json
import time
import thread
from multiprocessing import JoinableQueue

from messaging.messages.clip_eval_details import ClipEvalDetails
from messaging.messages.header import Header
from messaging.handlers.clip_ingest_handler import ClipIngestHandler
from messaging.infra.rpc_client import RpcClient
from messaging.infra.rpc_server import RpcServer

description = \
"""
Test script
"""

def startServer(amqp_url, exchangeName, listenRoutingKey, clipIngestQueue):
    clipIngestHandler = ClipIngestHandler(clipIngestQueue)
    rpc = RpcServer(amqp_url, exchangeName, listenRoutingKey, clipIngestHandler)

def process():
    amqp_url = 'localhost'
    exchangeName = 'samosa.khajuri_pipeline'

    clipIngestQueue = JoinableQueue()

    listenRoutingKey = 'samosa.khajuri_pipeline.clip_in.server.{}'.format(socket.gethostname())
    thread.start_new_thread(startServer, (amqp_url, exchangeName, listenRoutingKey, clipIngestQueue,))

    machineRoutingKey = 'samosa.khajuri_pipeline.result_out.server.{}'.format(socket.gethostname())
    responseRoutingKey = 'samosa.khajuri_pipeline.result_out.client.{}'.format(socket.gethostname())
    nimkiRpcClient = RpcClient(amqp_url, exchangeName, machineRoutingKey, responseRoutingKey)

    header = Header.dataRequest()

    while True:
        clipEvalMessage = clipIngestQueue.get()
        print "Working on clip: {}".format(clipEvalMessage.message['clipId'])
        if (clipEvalMessage.message['clipId'] == None) or (clipEvalMessage.message['clipId'] == ""):
            clipIngestQueue.task_done()
            break
        response = nimkiRpcClient.call(header.message, json.dumps(clipEvalMessage.message))
        time.sleep(10)
        clipIngestQueue.task_done()
        print "Finished working on clip: {}".format(clipEvalMessage.message['clipId'])

    # put poison pill back
    clipEvalMessage = ClipEvalDetails()
    response = nimkiRpcClient.call(header.message, json.dumps(clipEvalMessage.message))
    clipIngestQueue.join()

if __name__ == '__main__':
    process()

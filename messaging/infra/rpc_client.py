import pika
import uuid

class RpcClient(object):
    """RabbitMq based RPC Client"""

    def __init__(self, amqp_url, exchangeName, machineRoutingKey, responseRoutingKey=None):
        """Initialize"""
        self.amqp_url = amqp_url
        self.exchangeName = exchangeName
        self.machineRoutingKey = machineRoutingKey
        self.responseRoutingKey = responseRoutingKey
        self.expectReply = (self.responseRoutingKey != None)
        # start a new connection
        self.connect()

    def connect(self):
        """Initialize connection"""
        print "Re/Connecting to RabbitMq"
        # blocking connection allows us to avoid using callbacks in every step
        # set to large heatbeat duration so we don't have to re-connect every so often
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self.amqp_url, heartbeat_interval=(60 * 10))
        )
        self.channel = self.connection.channel()
        if self.expectReply:
            self.callback_queue = self.channel.queue_declare(exclusive=True).method.queue
            self.channel.queue_bind(
                exchange=self.exchangeName, queue=self.callback_queue,
                routing_key=self.responseRoutingKey
            )
            self.channel.basic_consume(self.on_response, queue=self.callback_queue)
        else:
            self.channel.queue_bind(exchange=self.exchangeName)

    def call(self, headers, message):
        """Publish helper"""
        if self.expectReply:
            return self.call_expectReply(headers, message)
        else:
            return self.call_noExpectReply(headers, message)

    def on_response(self, ch, method, props, body):
        """Consume callback"""
        if self.corr_id == props.correlation_id:
            self.response = body
            # ack the delivery of message
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def call_expectReply(self, headers, message):
        """Publish and expect reply"""
        self.response = None
        self.corr_id = str(uuid.uuid4())
        properties = pika.BasicProperties(
            reply_to=self.responseRoutingKey,
            correlation_id=self.corr_id,
            headers=headers,
            delivery_mode=1
        )
        self.publish(properties, message)
        # block until response
        while self.response is None:
            self.connection.process_data_events()
        return self.response

    def call_noExpectReply(self, headers, message):
        """Publish and don't expect reply"""
        properties = pika.BasicProperties(
            headers=headers,
            delivery_mode=1
        )
        self.publish(properties, message)
        return None

    def publish(self, properties, message):
        """Publish after reconnection if necessary"""
        try:
            self.basic_publish(properties, message)
        except:
            self.connect()
            self.basic_publish(properties, message)

    def basic_publish(self, properties, message):
        self.channel.basic_publish(
            exchange=self.exchangeName,
            routing_key=self.machineRoutingKey,
            properties=properties,
            body=message
        )

    def close(self):
        """Close connection"""
        self.connection.close()

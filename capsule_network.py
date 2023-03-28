import tensorflow as tf
from capsule_layers import ConvCapsuleLayer, DenseCapsuleLayer, Length

class CapsuleNetwork(tf.keras.models.Model):
    def __init__(self, input_size, n_class, routing_type):
        super(CapsuleNetwork, self).__init__()
        self.input_size = input_size
        self.n_class = n_class
        self.routing_type = routing_type

        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv_caps1 = ConvCapsuleLayer(num_capsules=32, capsule_dim=8, kernel_size=3, strides=2, padding='same')
        self.conv_caps2 = ConvCapsuleLayer(num_capsules=32, capsule_dim=16, kernel_size=3, strides=1, padding='same')
        self.dense_caps1 = DenseCapsuleLayer(num_capsules=32, capsule_dim=16)
        self.dense_caps2 = DenseCapsuleLayer(num_capsules=n_class, capsule_dim=16)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.out_caps = Length()

    def call(self, inputs):
        # Perform the forward pass
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.dense_caps1(x)

        if self.routing_type == 'None':
            # Perform the forward pass without dynamic routing
            x = self.dense_caps2(x)
            x = self.out_caps(x)
        else:
            # Perform the forward pass with dynamic routing
            batch_size = tf.shape(x)[0]
            capsule_dim = self.dense_caps2.capsule_dim
            num_capsules = self.dense_caps2.num_capsules

            # Initialize the logits and activation of the output capsules
            logits = tf.zeros([batch_size, num_capsules], dtype=tf.float32)
            activation = tf.zeros([batch_size, num_capsules, capsule_dim], dtype=tf.float32)

            # Perform dynamic routing
            num_routing = 3 if self.routing_type == 'EM' else 1

            if self.routing_type == 'FSA':
                # Compute the coupling coefficients using FSA algorithm
                for i in range(num_routing):
                    logits -= tf.reduce_sum(tf.square(activation), axis=-1, keepdims=True)
                    coupling_coefficients = tf.nn.softmax(logits, axis=1)
                    activation = tf.reduce_sum(tf.expand_dims(coupling_coefficients, axis=-1) * x, axis=1)
                    activation = activation / tf.norm(activation, axis=-1, keepdims=True)

            else:
                # Compute the coupling coefficients using EM algorithm
                for i in range(num_routing):
                    # Compute the coupling coefficients
                    coupling_coefficients = tf.nn.softmax(logits, axis=1)

                    # Compute the weighted sum of the predicted capsules
                    activation_weighted_sum = tf.reduce_sum(tf.expand_dims(coupling_coefficients, axis=-1) * x, axis=1)

                    # Apply the squash activation
                                    # Apply the squash activation
                    activation_norm = tf.norm(activation_weighted_sum, axis=-1, keepdims=True)
                    activation = activation_weighted_sum / (1 + activation_norm**2) * activation_norm / (1 + activation_norm)
    
                    # Update the logits using the predicted activations
                    if i < num_routing - 1:
                        delta_logits = tf.reduce_sum(x * activation, axis=-1)
                        logits += delta_logits
    
            # Compute the length of the output capsules to obtain the class probabilities
            x = self.dense_caps2(activation)
            x = self.out_caps(x)
    
        return x

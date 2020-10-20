# XCaps
A Model combines Xception and Dynamic Routing Layer of Capsule Networks in Keras (https://keras.io/)

This model is designed to combine advantages of XCeption and Capsule networks for brain tumor classification on publicly available MRI dataset shared in : https://figshare.com/articles/brain_tumor_dataset/1512427

In the original Xception architecture, after the global average pooling, instead of 2048-sized vectors and fully connected layer, the dynamic routing layer of the capsule architecture is used in this proposed XCaps architecture.

Capsule implementation was taken from  https://github.com/bojone/Capsule/ in 2019.

This model is firstly proposed in Ph.D. thesis "DETECTING BRAIN TUMORS USING DEEP LEARNING APPROACHES" in July 2020.

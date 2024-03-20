import tensorflow as tf

# 모델 경로 설정
model_path = './models'

# 모델 불러오기
model = tf.keras.models.load_model(model_path)

# TensorFlow Lite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TensorFlow Lite 모델 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

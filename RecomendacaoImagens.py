import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import matplotlib.pyplot as plt

# Função para carregar e pré-processar as imagens
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Adiciona a dimensão do batch
    image = image / 255.0  # Normaliza a imagem
    return image

# Função para carregar o dataset de imagens (em um diretório)
def load_dataset(image_folder):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    images = [load_and_preprocess_image(image_path) for image_path in image_paths]
    return np.vstack(images), image_paths

# Função para extrair as características das imagens usando ResNet50
def extract_features(images):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)  # Flatten
    return features

# Função para encontrar as imagens mais semelhantes usando a similaridade de cosseno
def find_similar_images(query_image, feature_vectors, image_paths, top_n=5):
    query_feature = extract_features(query_image)  # Extrai as características da imagem consultada
    similarities = cosine_similarity(query_feature, feature_vectors)
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]  # Pega os índices das imagens mais semelhantes
    return [image_paths[i] for i in similar_indices], similarities[0][similar_indices]

# Exemplo de uso
if __name__ == "__main__":
    # Carregar as imagens
    image_folder = 'path_to_your_images'  # Substitua pelo caminho do seu diretório de imagens
    images, image_paths = load_dataset(image_folder)

    # Extrair as características das imagens
    feature_vectors = extract_features(images)

    # Escolher uma imagem de consulta
    query_image_path = 'path_to_query_image.jpg'  # Substitua pelo caminho da imagem consultada
    query_image = load_and_preprocess_image(query_image_path)

    # Encontrar as imagens mais semelhantes
    similar_images, similarities = find_similar_images(query_image, feature_vectors, image_paths)

    # Exibir as imagens mais semelhantes
    print("Imagens semelhantes à consulta:")
    for i, (img_path, similarity) in enumerate(zip(similar_images, similarities)):
        print(f"Imagem {i+1}: {img_path} (similaridade: {similarity:.4f})")
        img = load_img(img_path, target_size=(224, 224))
        plt.subplot(1, len(similar_images), i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

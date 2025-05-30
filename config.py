# src/config.py

import torch
import os

# -------------------------
# Distributed Training (DDP) Configuration
# -------------------------
# Estas variables suelen ser gestionadas por el lanzador de DDP (torchrun o Slurm)
# y se establecen como variables de entorno.
# No necesitas definirlas manualmente aquí si usas torchrun.
# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1)) # Total de procesos (GPUs)
# RANK = int(os.environ.get("RANK", 0))             # Rank del proceso actual (0 a WORLD_SIZE-1)
# LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0)) # Rank local en el nodo (0 a GPUS_PER_NODE-1)
# MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost") # IP del nodo maestro
# MASTER_PORT = os.environ.get("MASTER_PORT", "29500")     # Puerto del nodo maestro

# Determina el dispositivo automáticamente (prioriza GPU si está disponible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
AMP_ENABLED = True
# -------------------------
# Paths Configuration
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Raíz del proyecto: confocal-cyclegan-project/
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
TRAIN_SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples_train")
EVAL_SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples_eval_2")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.npz")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.npz")

# Asegurarse de que los directorios de salida existan
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TRAIN_SAMPLES_DIR, exist_ok=True)
os.makedirs(EVAL_SAMPLES_DIR, exist_ok=True)

# -------------------------
# Dataset Configuration
# -------------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
# Imágenes monocromáticas (microscopía confocal en escala de grises)
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
# Normalización de imágenes al rango [-1, 1]
# (Valor medio y desviación estándar para la normalización)
# Dado que los datos están en [0, 255], normalizamos dividiendo por 127.5 y restando 1.
# Esto equivale a una transformación (x / 127.5) - 1
# Para torchvision.transforms.Normalize, necesitamos media y std
# Media = 1.0 (para llevar el centro de [0, 2] a 1)
# Std = 1.0 (para mantener el rango [0, 2] )
# Esto es conceptualmente incorrecto para Normalize.
# La forma correcta es usar ToTensor (escala a [0,1]) y luego Normalize(mean=[0.5], std=[0.5])
# O una lambda: transforms.Lambda(lambda x: (x / 127.5) - 1.0)
# Dejaremos la lambda en el dataset por claridad con el rango [-1, 1]
NORM_MEAN = [0.5] * INPUT_CHANNELS
NORM_STD = [0.5] * INPUT_CHANNELS

# -------------------------
# Model Architecture Configuration
# -------------------------
# Generador
GEN_TYPE = "resnet_12blocks" # Tipo de generador (podría ser "resnet_6blocks")
NGF = 64                    # Número de filtros en la primera capa convolucional del generador
USE_DROPOUT_GEN = False     # Usar dropout en los bloques ResNet del generador
INIT_TYPE_GEN = 'normal'    # Método de inicialización de pesos ('normal', 'xavier', 'kaiming', 'orthogonal')
INIT_GAIN_GEN = 0.02        # Ganancia para la inicialización
NORM_GEN = 'instance'
# Integración de Atención en el Generador
USE_ATTENTION = True        # Habilitar/deshabilitar módulos de atención en el generador
ATTENTION_TYPE = "CBAM"     # Tipo de atención ('CBAM', 'SelfAttention')

# Discriminador
DISC_TYPE = "patchgan"      # Tipo de discriminador
NDF = 64                    # Número de filtros en la primera capa convolucional del discriminador
N_LAYERS_DISC = 3           # Número de capas convolucionales en PatchGAN
INIT_TYPE_DISC = 'normal'   # Método de inicialización de pesos
INIT_GAIN_DISC = 0.02       # Ganancia para la inicialización
NORM_DISC = 'instance' 
# -------------------------
# Training Configuration
# -------------------------
NUM_EPOCHS = 120
START_EPOCH = 36            # Época desde la que empezar (útil si se reanuda el entrenamiento)
# Época a partir de la cual la tasa de aprendizaje empieza a decaer linealmente
EPOCH_DECAY_START = 45
# Batch size por GPU. El batch size total será BATCH_SIZE * WORLD_SIZE
BATCH_SIZE = 4
# Número de workers para el DataLoader por GPU
NUM_WORKERS = 8

# Optimizadores
LR_G = 0.0002               # Tasa de aprendizaje para los generadores
LR_D = 0.0002               # Tasa de aprendizaje para los discriminadores
BETA1 = 0.5                 # Parámetro beta1 para Adam
BETA2 = 0.999               # Parámetro beta2 para Adam

# Funciones de Pérdida
USE_VGG_LOSS = False
LAMBDA_VGG = 1.0
LOSS_MODE = 'lsgan'         # Tipo de pérdida adversarial ('lsgan' para MSELoss, 'vanilla' para BCEWithLogitsLoss)
LAMBDA_CYCLE = 15.0         # Peso para la pérdida de consistencia cíclica
LAMBDA_IDENTITY = 1.5       # Peso para la pérdida de identidad (0 para desactivar)

# Image Pool
POOL_SIZE = 50              # Tamaño del buffer de imágenes generadas para el entrenamiento del discriminador

# -------------------------
# Logging and Saving Configuration
# -------------------------
LOG_FREQ = 2000          # Frecuencia (en batches) para imprimir logs en consola
SAVE_EPOCH_FREQ = 5         # Frecuencia (en épocas) para guardar checkpoints del modelo
SAVE_LATEST_FREQ = 5000     # Frecuencia (en batches) para guardar el checkpoint 'latest'
SAVE_SAMPLES_EPOCH_FREQ = 1 # Frecuencia (en épocas) para guardar muestras visuales del entrenamiento

# Loss Tracking Configuration
ENABLE_LOSS_TRACKING = True  # Habilitar seguimiento y guardado de pérdidas
LOSS_PLOT_FREQ = 5          # Frecuencia (en épocas) para generar gráficos de pérdidas
LOSS_SAVE_FREQ = 1          # Frecuencia (en épocas) para guardar datos de pérdidas

# -------------------------
# Evaluation Configuration
# -------------------------
# Checkpoint a cargar para la evaluación (se completará con el nombre del archivo .pth)
EVAL_CHECKPOINT_G_A2B = "latest_netG_A2B.pth" # O un checkpoint específico como "200_netG_A2B.pth"
EVAL_CHECKPOINT_G_B2A = "latest_netG_B2A.pth" # Opcional si sólo evalúas A->B
EVAL_BATCH_SIZE = 4         # Batch size para la evaluación

# -------------------------
# Sanity Checks / Dynamic Adjustments (Opcional)
# -------------------------
# Ajustar el número de workers si es necesario
# NUM_WORKERS = min(NUM_WORKERS, os.cpu_count() // WORLD_SIZE if WORLD_SIZE > 0 else os.cpu_count())

print(f"--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Number of GPUs available: {NUM_GPUS}")
print(f"Batch size per GPU: {BATCH_SIZE}")
# print(f"Total batch size (if DDP): {BATCH_SIZE * WORLD_SIZE}") # Solo relevante si DDP está activo
print(f"Use Attention: {USE_ATTENTION}, Type: {ATTENTION_TYPE}")
print(f"Cycle Loss Lambda: {LAMBDA_CYCLE}")
print(f"Identity Loss Lambda: {LAMBDA_IDENTITY}")
print(f"---------------------")

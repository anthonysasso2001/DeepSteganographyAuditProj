# %% [markdown]
# ### Initial Database and Checks
#

# %%
from IPython.display import display
from tqdm.auto import tqdm
from IPython.display import clear_output
import shutil
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import words
import nltk
import pywt
from scipy import fft
from reedsolo import RSCodec
from dahuffman import HuffmanCodec
from LSB_Steganography.LSBSteg import LSBSteg
import string
import random
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import os
import tensorflow as tf
import gc

# NOTE: may have to change batch size and epochs depending on GPU VRAM. but epochs should be kept to 100 if possible for train accuracy.
# dataset vars
TRAIN_END = 15000
TEST_SIZE = int(TRAIN_END * 0.20)
HOLDOUT_END = TRAIN_END + TEST_SIZE
# secret text vars
MAX_CHARS = 20
DELIM = "|||END|||"
PAD_CHAR = "\x00"
# training vars
CLEAR_BEFORE_TRAIN = True
BATCH_SIZE = 64
STEPS = TRAIN_END//2 // BATCH_SIZE + 1
RS_BYTES = 8  # default; overridden per ablation
# --- Capacity note for 64×64×3 images ---
# Bottleneck: StatisticalSteganography(block_size=4) → 256 bits = 32 bytes (fixed_byte_len).
# After DELIM (9 B) + 2 B overhead, usable payload per RS_BYTES:
#   RS=4  → ~17 chars  (2-byte error correction)
#   RS=8  → ~13 chars  (4-byte error correction)   ← sensible default
#   RS=12 → ~9 chars   (6-byte error correction)
#   RS=16 → ~5 chars   (8-byte error correction)
# RS=32/64 eat all capacity and get clamped to 8 by determine_global_limits.
# Each dict defines one ablation run. Keys override the scalar defaults above.
ABLATION_CONFIGS = [
    # increase magnitude of loss by 3 for cover and secret to improve convergence but not skew cover/secret balance. also noise stats fairly early and peaks midway through with low rs.
    {"BETA": 3.0, "ALPHA": 3.0, "EPOCHS": 100, "LEARNING_RATE": 1e-3,
     "START_NOISE_EP": 10, "PEAK_NOISE_EP": 45, "RS_BYTES": 8},

    # version of best ablation but with no noise to test benefits of adversarial training
    {"BETA": 3.0, "ALPHA": 3.0, "EPOCHS": 100, "LEARNING_RATE": 1e-3,
     "START_NOISE_EP": 999, "PEAK_NOISE_EP": 999, "RS_BYTES": 8},

    # also create variations with the same 10%-45% noise ration for 50,150,200 epochs
    {"BETA": 3.0, "ALPHA": 3.0, "EPOCHS": 50, "LEARNING_RATE": 1e-3,
     "START_NOISE_EP": 5, "PEAK_NOISE_EP": 22, "RS_BYTES": 8},

    {"BETA": 3.0, "ALPHA": 3.0, "EPOCHS": 150, "LEARNING_RATE": 1e-3,
     "START_NOISE_EP": 15, "PEAK_NOISE_EP": 68, "RS_BYTES": 8},

    {"BETA": 3.0, "ALPHA": 3.0, "EPOCHS": 200, "LEARNING_RATE": 1e-3,
     "START_NOISE_EP": 20, "PEAK_NOISE_EP": 90, "RS_BYTES": 8},
]

# Define your checkpoint directory
checkpoint_dir = './checkpoints/'
models_dir = './models/'
data_dir = './data/'


# %%
tf.keras.backend.clear_session()

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# === GPU CHECK ===
print("=" * 60)
print("GPU/CUDA Configuration Check")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Devices Available: {len(physical_devices)}")
if physical_devices:
    for i, gpu in enumerate(physical_devices):
        print(f"  GPU {i}: {gpu}")
else:
    print("  ⚠️  WARNING: No GPUs detected! Training will be SLOW.")

# Display nvidia-smi output
print("\n" + "=" * 60)
print("NVIDIA GPU Status (nvidia-smi)")
print("=" * 60)
try:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
except Exception as e:
    print(f"Could not run nvidia-smi: {e}")
print("=" * 60 + "\n")


# Load Dataset
dataset = load_dataset("zh-plus/tiny-imagenet")
shuffled_train = dataset['train'].shuffle(seed=42)
shuffled_valid = dataset['valid'].shuffle(seed=42)

train_pool_pil = shuffled_train.select(range(TRAIN_END))['image']
test_pool_pil = shuffled_valid.select(range(TEST_SIZE))['image']
holdout_pool_pil = shuffled_train.select(
    range(TRAIN_END, HOLDOUT_END))['image']


def pil_to_np(pil_list):
    return np.array([np.array(img.convert('RGB'), dtype=np.uint8) for img in pil_list])


print("Converting pools to NumPy arrays...")
train_pool_np = pil_to_np(train_pool_pil)
test_pool_np = pil_to_np(test_pool_pil)
holdout_pool_np = pil_to_np(holdout_pool_pil)

print(
    f"Randomly sampled {len(train_pool_np)} train, {len(test_pool_np)} test, and {len(holdout_pool_np)} holdout images.")

train_cover_np, train_secret_np = train_test_split(
    train_pool_np, train_size=0.5, shuffle=True)
test_cover_np, test_secret_np = train_test_split(
    test_pool_np, train_size=0.5, shuffle=True)
holdout_cover_np, holdout_secret_np = train_test_split(
    holdout_pool_np, train_size=0.5, shuffle=True)


def to_scale(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img


train_dataset = tf.data.Dataset.from_tensor_slices((train_cover_np, train_secret_np)) \
    .shuffle(10000) \
    .batch(BATCH_SIZE) \
    .map(lambda c, s: (to_scale(c), to_scale(s)), num_parallel_calls=tf.data.AUTOTUNE) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

# TEST: Batch + Scale (No Shuffle)
test_dataset = tf.data.Dataset.from_tensor_slices((test_cover_np, test_secret_np)) \
    .batch(BATCH_SIZE) \
    .map(lambda c, s: (to_scale(c), to_scale(s)), num_parallel_calls=tf.data.AUTOTUNE) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

# HOLDOUT: Batch + Scale (No Shuffle)
holdout_dataset = tf.data.Dataset.from_tensor_slices((holdout_cover_np, holdout_secret_np)) \
    .map(lambda c, s: (to_scale(c), to_scale(s)), num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(tf.data.AUTOTUNE)

print(
    f"✓ Image conversion complete! ({len(train_dataset)} training pairs), ({len(test_dataset)} test pairs), ({len(holdout_dataset)} holdout pairs)")
# Clean up memory
del train_pool_pil, test_pool_pil, holdout_pool_pil, \
    train_pool_np, test_pool_np, holdout_pool_np, \
    train_cover_np, train_secret_np, test_cover_np, \
    test_secret_np, holdout_cover_np, holdout_secret_np


# %% [markdown]
# ### NN Arch and Setup
#

# %%

initializer = tf.keras.initializers.GlorotNormal(seed=12541)

# standard conv layer with relu but no normalization


def build_branch(inputs, kernel_size, name_prefix):
    x = layers.Conv2D(50, kernel_size, padding='same', activation='relu',
                      kernel_initializer=initializer, bias_initializer='zeros',
                      name=f'{name_prefix}_1')(inputs)
    x = layers.Conv2D(50, kernel_size, padding='same', activation='relu',
                      kernel_initializer=initializer, bias_initializer='zeros',
                      name=f'{name_prefix}_2')(x)
    x = layers.Conv2D(50, kernel_size, padding='same', activation='relu',
                      kernel_initializer=initializer, bias_initializer='zeros',
                      name=f'{name_prefix}_3')(x)
    x = layers.Conv2D(50, kernel_size, padding='same', activation='relu',
                      kernel_initializer=initializer, bias_initializer='zeros',
                      name=f'{name_prefix}_4')(x)
    return x

# above but also use batch norm and residual to improve outcome of reveal network


def build_stable_branch(inputs, kernel_size, name_prefix):
    shortcut = inputs
    x = inputs

    for i in range(4):
        x = layers.Conv2D(
            50, kernel_size, padding='same', kernel_initializer=initializer, bias_initializer='zeros',
            name=f'{name_prefix}_{i+1}_conv'
        )(x)

        x = layers.BatchNormalization(
            name=f'{name_prefix}_{i+1}_bn'
        )(x)

        x = layers.Activation('relu',
                              name=f'{name_prefix}_{i+1}_relu'
                              )(x)

    # Match channels if needed
    if shortcut.shape[-1] != 50:
        shortcut = layers.Conv2D(
            50, (1, 1), padding='same', kernel_initializer=initializer,
            name=f'{name_prefix}_proj'
        )(shortcut)

    x = layers.Add(name=f'{name_prefix}_residual_add')([x, shortcut])

    return x


def build_keras_prep_network(input_shape=(64, 64, 3)):
    inputs = layers.Input(shape=input_shape, name="secret_image")

    # --- 3x3, 4x4, 5x5 Branches ---
    p3 = build_branch(inputs, (3, 3), "prep_conv3x3")
    p4 = build_branch(inputs, (4, 4), "prep_conv4x4")
    p5 = build_branch(inputs, (5, 5), "prep_conv5x5")

    # --- Concatenation (150 channels) ---
    concat_1 = layers.Concatenate(axis=3, name="prep_concat_1")([p3, p4, p5])

    # --- Final Processing (Layer 5) ---
    # These take concat_1 as input to fuse the multi-scale features
    p3_f = layers.Conv2D(50, (3, 3), padding='same', activation='relu',
                         kernel_initializer=initializer, bias_initializer='zeros',
                         name='prep_conv3x3_5')(concat_1)
    p4_f = layers.Conv2D(50, (4, 4), padding='same', activation='relu',
                         kernel_initializer=initializer, bias_initializer='zeros',
                         name='prep_conv4x4_5')(concat_1)
    p5_f = layers.Conv2D(50, (5, 5), padding='same', activation='relu',
                         kernel_initializer=initializer, bias_initializer='zeros',
                         name='prep_conv5x5_5')(concat_1)

    outputs = layers.Concatenate(
        axis=3, name="prep_concat_final")([p3_f, p4_f, p5_f])

    return Model(inputs=inputs, outputs=outputs, name="Keras_PrepNetwork")


def build_keras_hide_network(input_shape=(64, 64, 3), prep_channels=150):
    # Inputs
    cover_input = layers.Input(shape=input_shape, name="cover_input")
    prep_input = layers.Input(shape=(64, 64, prep_channels), name="prep_input")

    # Concatenate Cover and Prep (3 + 150 = 153 channels)
    concat_1 = layers.Concatenate(axis=3, name="hide_concat_1")([
        cover_input, prep_input])

    # --- 3x3, 4x4, 5x5 Branches ---

    h3 = build_branch(concat_1, (3, 3), "hide_conv3x3")
    h4 = build_branch(concat_1, (4, 4), "hide_conv4x4")
    h5 = build_branch(concat_1, (5, 5), "hide_conv5x5")

    # --- Second Concatenation (150 channels) ---
    concat_2 = layers.Concatenate(axis=3, name="hide_concat_2")([h3, h4, h5])

    # --- Final Branch Processing ---
    h3_f = layers.Conv2D(50, (3, 3), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='hide_conv3x3_5')(concat_2)
    h4_f = layers.Conv2D(50, (4, 4), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='hide_conv4x4_5')(concat_2)
    h5_f = layers.Conv2D(50, (5, 5), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='hide_conv5x5_5')(concat_2)

    concat_final = layers.Concatenate(
        axis=3, name="hide_concat_final")([h3_f, h4_f, h5_f])

    # add residual pass to improve performance and hopefully improve overall results
    concat_final = layers.Add(name="hide_residual")([concat_final, concat_2])

    # --- Output (Stego Image) ---
    # Using Sigmoid to ensure pixel values are between 0 and 1
    output = layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid',
                           kernel_initializer=initializer, bias_initializer='zeros', name='stego_output')(concat_final)

    return Model(inputs=[cover_input, prep_input], outputs=output, name="Keras_HideNetwork")


def build_keras_reveal_network(input_shape=(64, 64, 3)):
    stego_input = layers.Input(shape=input_shape, name="stego_input")

    # --- 3x3, 4x4, 5x5 Branches ---

    r3 = build_stable_branch(stego_input, (3, 3), "reveal_conv3x3")
    r4 = build_stable_branch(stego_input, (4, 4), "reveal_conv4x4")
    r5 = build_stable_branch(stego_input, (5, 5), "reveal_conv5x5")

    # --- Concatenation (150 channels) ---
    concat_1 = layers.Concatenate(axis=3, name="reveal_concat_1")([r3, r4, r5])

    # --- Final Processing ---
    r3_f = layers.Conv2D(50, (3, 3), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='reveal_conv3x3_5')(concat_1)
    r4_f = layers.Conv2D(50, (4, 4), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='reveal_conv4x4_5')(concat_1)
    r5_f = layers.Conv2D(50, (5, 5), padding='same', activation=None,
                         kernel_initializer=initializer, bias_initializer='zeros', name='reveal_conv5x5_5')(concat_1)

    concat_final = layers.Concatenate(
        axis=3, name="reveal_concat_final")([r3_f, r4_f, r5_f])

    # --- Output (Revealed Secret) ---
    output = layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid',
                           kernel_initializer=initializer, bias_initializer='zeros', name='revealed_secret')(concat_final)

    return Model(inputs=stego_input, outputs=output, name="Keras_RevealNetwork")


# %%


def steganography_loss(cover_input, secret_input, cover_output, secret_output, alpha=1.0, beta=10.0):
    c_in = tf.cast(cover_input, tf.float32)
    s_in = tf.cast(secret_input, tf.float32)
    c_out = tf.cast(cover_output, tf.float32)
    s_out = tf.cast(secret_output, tf.float32)

    # Calculate individual MSE
    cover_mse = tf.reduce_mean(tf.square(c_in - c_out))
    cover_log_loss = -tf.math.log(1.0 - cover_mse + 1e-7)

    # Adding 1e-7 prevents log(0) which would return NaN (hopefuly better than bce)
    secret_mse = tf.reduce_mean(tf.square(s_in - s_out))
    secret_log_loss = -tf.math.log(1.0 - secret_mse + 1e-7)
    # changing to bce to hopefully make fidelity better
    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # secret_bce = bce(s_in, s_out)

    # Total loss
    total_loss = (tf.cast(alpha, tf.float32) * cover_log_loss) + \
        (tf.cast(beta, tf.float32) * secret_log_loss)

    return total_loss, cover_mse, secret_log_loss


def compute_max_chars(capacity_bytes, rs_bytes, delim_len):
    max_chars = capacity_bytes - rs_bytes - delim_len - 2
    return max(1, max_chars)


class StegoSystem(tf.keras.Model):
    def __init__(self, prep_net, hide_net, reveal_net, stego_tools,
                 steps_per_epoch, word_list=None, max_safe_chars=12,
                 alpha=1.0, beta=10.0,
                 noise_start_epoch=10, noise_peak_epoch=40):
        super(StegoSystem, self).__init__()

        self.prep_net = prep_net
        self.hide_net = hide_net
        self.reveal_net = reveal_net
        self.stego_tools = stego_tools

        if word_list:
            self._words_by_len = {}
            for w in word_list:
                self._words_by_len.setdefault(len(w), []).append(w)
            self._word_len_keys = sorted(self._words_by_len)
        else:
            self._words_by_len = None
            self._word_len_keys = []
        self.max_safe_chars = max(4, max_safe_chars)
        self.beta = beta
        self.alpha = alpha

        self.methods = ['none', 'dct',
                        'dwt', 'spread_spectrum', 'statistical']

        # Curriculum schedule (step-based)
        self.noise_start_step = tf.Variable(
            float(noise_start_epoch * steps_per_epoch),
            trainable=False, dtype=tf.float32
        )
        self.noise_peak_step = tf.Variable(
            float(noise_peak_epoch * steps_per_epoch),
            trainable=False, dtype=tf.float32
        )

        # Metrics
        self.method_psnr = [
            tf.keras.metrics.Mean(name=f"psnr_{m}")
            for m in self.methods
        ]

        self.method_ssim = [
            tf.keras.metrics.Mean(name=f"ssim_{m}")
            for m in self.methods
        ]

        self.psnr_c_tracker = tf.keras.metrics.Mean(name="cover_psnr")
        self.ssim_s_tracker = tf.keras.metrics.Mean(name="secret_ssim")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.c_loss_tracker = tf.keras.metrics.Mean(name="cover_loss")
        self.s_loss_tracker = tf.keras.metrics.Mean(name="secret_loss")
        self.noise_prob_metric = tf.keras.metrics.Mean(name="noise_prob")
        self.payload_len_tracker = tf.keras.metrics.Mean(name="payload_len")

    @property
    def metrics(self):
        base = [
            self.total_loss_tracker,
            self.psnr_c_tracker,
            self.ssim_s_tracker,
            self.c_loss_tracker,
            self.s_loss_tracker,
            self.noise_prob_metric,
            self.payload_len_tracker,
        ]

        # + list(self.method_embed_success)
        return base + list(self.method_psnr) + list(self.method_ssim)

    def _apply_stego_numpy(self, secret_batch, method_idx, progress):
        batch = secret_batch.numpy()
        method_idx_val = int(method_idx.numpy())
        method = self.methods[method_idx_val]
        progress_val = float(np.clip(progress.numpy(), 0.0, 1.0))

        if method == 'none':
            return (
                batch.astype(np.float32),
                np.float32(0.0),
                np.array([method_idx_val], dtype=np.int32),
                np.float32(0.0),
                np.float32(0.0)
            )

        batch_uint8 = (batch * 255).astype(np.uint8)
        stego_batch = []

        tool = self.stego_tools[method]
        codec = self.stego_tools['codec']
        delim_len = len(DELIM)

        payload_lengths = []
        success_flags = []

        for img in batch_uint8:
            success = False

            # Capacity in bits from method and image shape
            capacity_bits = int(tool.get_capacity(img.shape))
            capacity_bytes = max(1, capacity_bits // 8)

            # Keep payload creation consistent with each tool's actual embed path
            tool_expected_len = getattr(tool, 'expected_len', None)
            effective_expected_len = int(
                tool_expected_len) if tool_expected_len is not None else capacity_bytes
            effective_expected_len = max(
                1, min(effective_expected_len, capacity_bytes))

            if effective_expected_len <= (RS_BYTES + delim_len + 2):
                stego_batch.append(img)
                payload_lengths.append(0)
                success_flags.append(0.0)
                continue

            max_chars = compute_max_chars(
                effective_expected_len,
                RS_BYTES,
                delim_len
            )

            # Curriculum scaling
            curriculum_max = max(1, int(progress_val * max_chars))

            for _ in range(2):

                # Beta distribution to bias toward larger payloads as curriculum progresses
                sampled = np.random.beta(2, 1)

                random_len = int(np.ceil(sampled * curriculum_max))

                # ✅ Clamp to valid range
                random_len = max(1, min(random_len, max_chars))

                if self._words_by_len:
                    # Pick the largest available bucket <= curriculum_max
                    valid_keys = [
                        k for k in self._word_len_keys if k <= random_len]
                    if not valid_keys:
                        valid_keys = [self._word_len_keys[0]]
                    bucket_len = valid_keys[-1]
                    train_text = random.choice(self._words_by_len[bucket_len])
                else:
                    train_text = ''.join(random.choices(
                        string.ascii_letters + string.digits,
                        k=random_len
                    ))

                try:
                    bits = prepare_payload(
                        train_text,
                        codec,
                        expected_len=effective_expected_len
                    )

                    if len(bits) <= capacity_bits:
                        s_img = tool.embed(img, train_text, codec)
                        stego_batch.append(s_img)
                        payload_lengths.append(random_len)
                        success_flags.append(1.0)
                        success = True
                        break

                except Exception:
                    continue

            if not success:
                stego_batch.append(img)
                payload_lengths.append(0)
                success_flags.append(0.0)

        avg_len = np.float32(np.mean(payload_lengths)
                             if payload_lengths else 0.0)
        avg_success = np.float32(
            np.mean(success_flags) if success_flags else 0.0)
        attempted_flag = np.float32(1.0)

        return (
            (np.array(stego_batch) / 255.0).astype(np.float32),
            avg_len,
            # Uses the defined variable
            np.array([method_idx_val], dtype=np.int32),
            avg_success,
            attempted_flag
        )

    def train_step(self, data):
        cover, secret = data
        current_step = tf.cast(self.optimizer.iterations, tf.float32)

        # Smooth curriculum progression
        den = tf.maximum(self.noise_peak_step - self.noise_start_step, 1.0)
        raw_prob = (current_step - self.noise_start_step) / den

        prob_noise = tf.clip_by_value(raw_prob, 0.0, 1.0)
        self.noise_prob_metric.update_state(prob_noise)

        progress = prob_noise  # shared curriculum signal

        should_augment = tf.less(
            tf.random.uniform([], 0, 1), prob_noise
        )

        method_idx = tf.cond(
            should_augment,
            lambda: tf.random.uniform(
                [], 1, len(self.methods), dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )

        with tf.GradientTape() as tape:
            augmented_secret, avg_len, method_idx_out, avg_success, attempted_flag = tf.py_function(
                func=self._apply_stego_numpy,
                inp=[secret, method_idx, progress],
                Tout=[tf.float32, tf.float32, tf.int32, tf.float32, tf.float32]
            )

            augmented_secret.set_shape(secret.shape)
            avg_len.set_shape(())
            method_idx_out.set_shape((1,))
            avg_success.set_shape(())
            attempted_flag.set_shape(())
            self.payload_len_tracker.update_state(avg_len)

            p_out = self.prep_net(augmented_secret, training=True)
            h_out = self.hide_net([cover, p_out], training=True)
            r_out = self.reveal_net(h_out, training=True)

            total_loss, c_loss, s_loss = steganography_loss(
                cover, augmented_secret, h_out, r_out,
                self.alpha, self.beta
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Metrics
        cover_f = tf.cast(cover, tf.float32)
        h_out_f = tf.cast(h_out, tf.float32)
        secret_f = tf.cast(augmented_secret, tf.float32)
        r_out_f = tf.cast(r_out, tf.float32)

        psnr_val = tf.reduce_mean(tf.image.psnr(cover_f, h_out_f, max_val=1.0))
        ssim_val = tf.reduce_mean(
            tf.image.ssim(secret_f, r_out_f, max_val=1.0))

        self.psnr_c_tracker.update_state(psnr_val)
        self.ssim_s_tracker.update_state(ssim_val)

        # ✅ Tensor-safe method indexing
        method_idx_scalar = tf.cast(method_idx_out[0], tf.int32)

        tf.switch_case(
            method_idx_scalar,
            branch_fns=[
                lambda i=i: self.method_psnr[i].update_state(psnr_val)
                for i in range(len(self.methods))
            ]
        )

        tf.switch_case(
            method_idx_scalar,
            branch_fns=[
                lambda i=i: self.method_ssim[i].update_state(ssim_val)
                for i in range(len(self.methods))
            ]
        )

        self.total_loss_tracker.update_state(total_loss)
        self.c_loss_tracker.update_state(c_loss)
        self.s_loss_tracker.update_state(s_loss)

        results = {m.name: m.result() for m in self.metrics}
        results["lr"] = self.optimizer.learning_rate

        return results

    def test_step(self, data):
        cover, secret = data
        current_step = tf.cast(self.optimizer.iterations, tf.float32)

        should_augment = tf.greater_equal(
            current_step, self.noise_start_step)

        noise_val = tf.cond(should_augment, lambda: 1.0, lambda: 0.0)
        self.noise_prob_metric.update_state(noise_val)

        progress = tf.constant(1.0)

        method_idx = tf.cond(
            should_augment,
            lambda: tf.random.uniform(
                [], 1, len(self.methods), dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )

        # 🔧 IMPORTANT: match train_step outputs
        augmented_secret, avg_len, method_idx_out, avg_success, attempted_flag = tf.py_function(
            func=self._apply_stego_numpy,
            inp=[secret, method_idx, progress],
            Tout=[tf.float32, tf.float32, tf.int32, tf.float32, tf.float32]
        )

        augmented_secret.set_shape(secret.shape)
        avg_len = tf.reshape(avg_len, [])
        method_idx_out.set_shape((1,))
        avg_success = tf.reshape(avg_success, [])
        attempted_flag = tf.reshape(attempted_flag, [])

        self.payload_len_tracker.update_state(avg_len)

        # Forward pass
        p_out = self.prep_net(augmented_secret, training=False)
        h_out = self.hide_net([cover, p_out], training=False)
        r_out = self.reveal_net(h_out, training=False)

        total_loss, c_loss, s_loss = steganography_loss(
            cover, augmented_secret, h_out, r_out,
            self.alpha, self.beta
        )

        # 🔥 ADD THIS BLOCK (missing piece)
        cover_f = tf.cast(cover, tf.float32)
        h_out_f = tf.cast(h_out, tf.float32)
        secret_f = tf.cast(augmented_secret, tf.float32)
        r_out_f = tf.cast(r_out, tf.float32)

        psnr_val = tf.reduce_mean(tf.image.psnr(cover_f, h_out_f, max_val=1.0))
        ssim_val = tf.reduce_mean(
            tf.image.ssim(secret_f, r_out_f, max_val=1.0))

        self.psnr_c_tracker.update_state(psnr_val)
        self.ssim_s_tracker.update_state(ssim_val)

        # Optional: per-method tracking (same as train_step)
        method_idx_scalar = tf.cast(method_idx_out[0], tf.int32)

        tf.switch_case(
            method_idx_scalar,
            branch_fns=[
                lambda i=i: self.method_psnr[i].update_state(psnr_val)
                for i in range(len(self.methods))
            ]
        )

        tf.switch_case(
            method_idx_scalar,
            branch_fns=[
                lambda i=i: self.method_ssim[i].update_state(ssim_val)
                for i in range(len(self.methods))
            ]
        )

        # Loss metrics
        self.total_loss_tracker.update_state(total_loss)
        self.c_loss_tracker.update_state(c_loss)
        self.s_loss_tracker.update_state(s_loss)

        return {m.name: m.result() for m in self.metrics}


# %% [markdown]
# ### Loss func and Training Loop
#

# %%
# Prep lsb / huffman code


def qim_embed(x, bit, delta):
    """
    Embeds a bit (0 or 1) into coefficient x using a quantization step (delta).
    Bit 0 quantizes to even multiples, Bit 1 quantizes to odd multiples.
    """
    return np.round((x - bit * (delta / 2.0)) / delta) * delta + bit * (delta / 2.0)


def qim_extract(x, delta):
    """
    Extracts a bit from coefficient x by measuring distance to the nearest grid.
    """
    dist0 = np.abs(x - np.round(x / delta) * delta)
    dist1 = np.abs(
        x - (np.round((x - delta / 2.0) / delta) * delta + delta / 2.0))
    return (dist1 < dist0).astype(np.uint8)

# --- Binary Conversions ---


def bytes_to_bits(byte_data):
    return np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))


def bits_to_bytes(bit_array):
    return np.packbits(bit_array).tobytes()


def prepare_payload(text, codec, expected_len):
    """
    Safe payload builder:
    - No silent truncation
    - Clean zero padding
    - RS encoding applied last
    """

    # 1. Add delimiter
    marked_text = text + DELIM

    # 2. Huffman encode
    encoded = codec.encode(marked_text)

    max_payload_bytes = expected_len - RS_BYTES

    # ❌ DO NOT TRUNCATE — HARD FAIL INSTEAD
    if len(encoded) > max_payload_bytes:
        raise ValueError(
            f"Encoded payload too large ({len(encoded)} bytes > {max_payload_bytes})"
        )

    # ✅ SAFE padding (zero padding ONLY)
    padded = encoded.ljust(max_payload_bytes, b'\x00')

    # 3. RS encode
    ecc_bytes = rs.encode(padded)

    # 4. Convert to bits
    return np.unpackbits(np.frombuffer(ecc_bytes, dtype=np.uint8))


def decode_payload(bit_array, codec, expected_len):
    try:
        raw_bytes = np.packbits(bit_array).tobytes()[:expected_len]
        repaired = rs.decode(raw_bytes)[0]

        repaired = repaired.rstrip(b'\x00')
        if not repaired:
            return ""

        decoded = codec.decode(repaired)
        if isinstance(decoded, bytes):
            decoded = decoded.decode('utf-8', errors='ignore')

        end_idx = decoded.find(DELIM)

        if end_idx != -1:
            return decoded[:end_idx]

        return ""

    except Exception:
        return ""


class LSBSteganography:
    def __init__(self, use_header=True, expected_len=None):
        self.use_header = use_header
        self.expected_len = expected_len
        if not use_header and expected_len is None:
            raise ValueError(
                "expected_len must be provided if use_header is False.")

    def get_capacity(self, image_shape=(64, 64, 3)):
        # 3 bits per pixel (RGB)
        return image_shape[0] * image_shape[1] * 3

    def embed(self, image_array, text, codec):
        bit_array = prepare_payload(
            text, codec, expected_len=self.expected_len
        )

        # ✅ FIX: capacity in BITS
        max_bits = image_array.shape[0] * image_array.shape[1] * 3

        if len(bit_array) > max_bits:
            raise ValueError(
                f"Payload ({len(bit_array)} bits) exceeds LSB capacity ({max_bits} bits)!"
            )

        data_to_hide = bits_to_bytes(bit_array)

        try:
            steg = LSBSteg(image_array)
            return steg.encode_binary(data_to_hide)
        except Exception as e:
            print(f"LSB Encoding internal error: {e}")
            return image_array

    def extract(self, image_array, codec):
        try:
            steg = LSBSteg(image_array)
            # 4. Extract raw binary.
            # Note: LSBSteg.decode_binary() returns the entire bitstream of the image
            raw_binary = steg.decode_binary()

            # Convert the raw byte string back into a bit array for our decoder
            # This ensures it matches the format expected by decode_payload
            raw_bits = bytes_to_bits(raw_binary)

            return decode_payload(raw_bits, codec, expected_len=self.expected_len)
        except Exception as e:
            # If the image is too distorted for the library to find LSBs
            return ""


class DWTSteganography:
    def __init__(self, delta=25.0, band='LH', use_header=True, expected_len=None, rep=1):
        self.delta = delta
        self.band = band.upper()
        self.use_header = use_header
        self.expected_len = expected_len
        self.rep = max(1, int(rep))

    def get_capacity(self, image_shape=(64, 64, 3)):
        # 1 bit per coefficient in selected band (rep reduces usable capacity)
        return (image_shape[0] // 2) * (image_shape[1] // 2) // self.rep

    def embed(self, image_array, text, codec):
        img_float = image_array.astype(np.float32)
        bit_array = prepare_payload(
            text, codec, expected_len=self.expected_len)

        coeffs = pywt.dwt2(img_float, 'haar', axes=(0, 1))
        LL, (LH, HL, HH) = coeffs
        bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
        target = bands[self.band]
        flat_band = target.flatten()

        if len(bit_array) * self.rep > len(flat_band):
            raise ValueError(
                f"Payload with rep={self.rep} exceeds DWT {self.band} capacity!")

        # Fill remainder with random noise, then stamp each bit rep times
        padded_bits = np.random.randint(
            0, 2, size=len(flat_band), dtype=np.uint8)
        r = self.rep
        for i, b in enumerate(bit_array):
            padded_bits[i * r:(i + 1) * r] = b
        flat_band = qim_embed(flat_band, padded_bits, self.delta)

        new_target = flat_band.reshape(target.shape)
        if self.band == 'LL':
            LL = new_target
        elif self.band == 'LH':
            LH = new_target
        elif self.band == 'HL':
            HL = new_target
        else:
            HH = new_target

        stego = pywt.idwt2((LL, (LH, HL, HH)), 'haar', axes=(0, 1))
        return np.clip(stego, 0, 255).astype(np.uint8)

    def extract(self, image_array, codec):
        img_float = image_array.astype(np.float32)
        LL, (LH, HL, HH) = pywt.dwt2(img_float, 'haar', axes=(0, 1))
        bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}

        raw_bits = qim_extract(bands[self.band].flatten(), self.delta)
        r = self.rep
        if r > 1:
            num_bits = self.expected_len * 8
            # Majority vote across rep copies of each bit
            extracted_bits = np.array([
                int(round(np.mean(raw_bits[i * r:(i + 1) * r])))
                for i in range(num_bits)
            ], dtype=np.uint8)
        else:
            extracted_bits = raw_bits
        return decode_payload(extracted_bits, codec, expected_len=self.expected_len)


class GridDCTSteganography:
    def __init__(self, delta=64.0, block_size=4, use_header=False, expected_len=None, rep=1):
        self.delta = delta
        self.block_size = block_size
        self.use_header = use_header
        self.expected_len = expected_len
        self.rep = max(1, int(rep))

    def get_capacity(self, image_shape=(64, 64, 3)):
        # Total blocks / rep = effective payload bits
        return 3 * (image_shape[0] // self.block_size) * (image_shape[1] // self.block_size) // self.rep

    def embed(self, image_array, text, codec):
        img_float = image_array.astype(np.float32)
        bit_array = prepare_payload(
            text, codec, expected_len=self.expected_len)

        H, W, C = img_float.shape
        rows, cols = H // self.block_size, W // self.block_size

        # Each payload bit is written into `rep` consecutive blocks
        expanded_bits = np.repeat(bit_array, self.rep)
        bit_idx = 0
        for c in range(3):  # Iterate Red, Green, Blue
            channel = img_float[:, :, c]
            for i in range(rows):
                for j in range(cols):
                    if bit_idx < len(expanded_bits):
                        y, x = i * self.block_size, j * self.block_size
                        block = channel[y:y+self.block_size,
                                        x:x+self.block_size]

                        dct_block = fft.dctn(block, norm='ortho')
                        target_idx = self.block_size // 2
                        coeff = dct_block[target_idx, target_idx]

                        # Quantization Logic
                        shift = 0.75 if expanded_bits[bit_idx] == 1 else 0.25
                        coeff = np.floor(coeff / self.delta) * \
                            self.delta + (shift * self.delta)

                        dct_block[target_idx, target_idx] = coeff
                        channel[y:y+self.block_size, x:x +
                                self.block_size] = fft.idctn(dct_block, norm='ortho')

                        bit_idx += 1
            img_float[:, :, c] = channel

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def extract(self, image_array, codec):
        img_float = image_array.astype(np.float32)
        H, W, C = img_float.shape
        rows, cols = H // self.block_size, W // self.block_size

        # How many payload bits to reconstruct
        if self.use_header:
            num_bits = self.get_capacity(image_array.shape)
        else:
            num_bits = self.expected_len * 8

        # Read rep copies of each bit then majority vote
        num_to_read = num_bits * self.rep
        extracted_raw = []
        bit_idx = 0
        for c in range(3):
            channel = img_float[:, :, c]
            for i in range(rows):
                for j in range(cols):
                    if bit_idx < num_to_read:
                        y, x = i * self.block_size, j * self.block_size
                        block = channel[y:y+self.block_size,
                                        x:x+self.block_size]
                        dct_block = fft.dctn(block, norm='ortho')

                        target_idx = self.block_size // 2
                        coeff = dct_block[target_idx, target_idx]

                        val = (coeff % self.delta) / self.delta
                        extracted_raw.append(1 if val > 0.5 else 0)
                        bit_idx += 1

        if self.rep > 1:
            extracted_bits = np.array([
                int(round(
                    np.mean(extracted_raw[i * self.rep:(i + 1) * self.rep])))
                for i in range(num_bits)
            ], dtype=np.uint8)
        else:
            extracted_bits = np.array(extracted_raw)
        return decode_payload(extracted_bits, codec, expected_len=self.expected_len)


class SpreadSpectrumSteganography:
    def __init__(self, gain=100.0, max_bits=None, use_header=False, expected_len=None):
        self.gain = gain
        # If max_bits isn't provided, default to the byte length * 8
        self.max_bits = max_bits if max_bits is not None else (
            expected_len * 8 if expected_len else 256)
        self.use_header = use_header
        self.expected_len = expected_len

    def get_capacity(self, image_shape=(64, 64, 3)):
        return image_shape[0] * image_shape[1] * image_shape[2]

    def embed(self, image_array, text, codec):
        img_float = image_array.astype(np.float32)
        actual_bits = prepare_payload(
            text, codec, expected_len=self.expected_len)

        # Pad to fixed capacity so chips_per_bit never changes
        if len(actual_bits) > self.max_bits:
            raise ValueError(
                f"Payload too large for Spread Spectrum ({len(actual_bits)} > {self.max_bits})"
            )

        padding = np.zeros(self.max_bits - len(actual_bits), dtype=np.uint8)
        bit_array = np.concatenate([actual_bits, padding])

        bipolar_bits = 2 * bit_array.astype(np.float32) - 1
        flat_img = img_float.flatten()
        np.random.seed(42)
        carrier = np.random.normal(0, 1, len(flat_img))

        chips_per_bit = len(flat_img) // self.max_bits
        signal = np.repeat(bipolar_bits, chips_per_bit)
        full_signal = np.zeros_like(flat_img)
        full_signal[:len(signal)] = signal * carrier[:len(signal)]

        return np.clip(flat_img + (self.gain * full_signal), 0, 255).reshape(image_array.shape).astype(np.uint8)

    def extract(self, image_array, codec):
        img_float = image_array.astype(np.float32).flatten()
        # FIX: Mean-center to remove image bias
        img_centered = img_float - np.mean(img_float)

        np.random.seed(42)
        carrier = np.random.normal(0, 1, len(img_float))
        chips_per_bit = len(img_float) // self.max_bits

        extracted_bits = []
        for b in range(self.max_bits):
            start = b * chips_per_bit
            end = start + chips_per_bit
            correlation = np.sum(img_centered[start:end] * carrier[start:end])
            extracted_bits.append(1 if correlation > 0 else 0)

        return decode_payload(extracted_bits, codec, expected_len=self.expected_len)


class StatisticalSteganography:
    def __init__(self, block_size=4, threshold=30.0, use_header=False, expected_len=None):
        self.block_size = block_size
        self.threshold = threshold
        self.use_header = use_header
        self.expected_len = expected_len

    def get_capacity(self, image_shape=(64, 64, 3)):
        # 1 bit per block_size x block_size area
        return (image_shape[0] // self.block_size) * (image_shape[1] // self.block_size)

    def embed(self, image_array, text, codec):
        img_float = image_array.astype(np.float32)
        actual_bits = prepare_payload(
            text, codec, expected_len=self.expected_len)
        H, W, _ = img_float.shape
        total_blocks = (H // self.block_size) * (W // self.block_size)

        # Pad payload to fill all blocks
        if len(actual_bits) > total_blocks:
            raise ValueError(
                f"Payload too large for Statistical ({len(actual_bits)} > {total_blocks})"
            )

        padding = np.zeros(total_blocks - len(actual_bits), dtype=np.uint8)
        bit_array = np.concatenate([actual_bits, padding])

        channel = img_float[:, :, 0]
        idx = 0
        for y in range(0, H - self.block_size + 1, self.block_size):
            for x in range(0, W - self.block_size + 1, self.block_size):
                if idx < len(bit_array):
                    block = channel[y:y+self.block_size, x:x+self.block_size]
                    mid = self.block_size // 2
                    val = self.threshold if bit_array[idx] == 1 else - \
                        self.threshold
                    block[:mid, :] += val
                    block[mid:, :] -= val
                    idx += 1
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def extract(self, image_array, codec):
        img_float = image_array.astype(np.float32)
        channel = img_float[:, :, 0]
        H, W, _ = img_float.shape
        total_blocks = (H // self.block_size) * (W // self.block_size)

        extracted_bits = []
        for y in range(0, H - self.block_size + 1, self.block_size):
            for x in range(0, W - self.block_size + 1, self.block_size):
                block = channel[y:y+self.block_size, x:x+self.block_size]
                mid = self.block_size // 2
                extracted_bits.append(1 if np.mean(
                    block[:mid, :]) > np.mean(block[mid:, :]) else 0)

        return decode_payload(extracted_bits[:total_blocks], codec, expected_len=self.expected_len)


def find_max_supported_word(word_list, codec, fixed_byte_len, use_header=False):
    """
    Finds the longest word in the list that fits within the bit-budget.
    Returns the word and its bit length.
    """
    max_bits = fixed_byte_len * 8
    supported_words = []

    for word in word_list:
        try:
            bits = prepare_payload(word, codec,
                                   expected_len=fixed_byte_len)

            if len(bits) <= max_bits:
                supported_words.append((word, len(bits)))
        except Exception:
            continue  # Skip words with characters not in your codec

    if not supported_words:
        return None, 0

    # Sort by character length, then bit length
    supported_words.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)

    best_word, bit_size = supported_words[0]
    print(f"✅ Max Capacity Check:")
    print(f"   - Target Bit Budget: {max_bits} bits ({fixed_byte_len} bytes)")
    print(
        f"   - Longest Supported Word: '{best_word}' ({len(best_word)} chars)")
    print(
        f"   - Bit Utilization: {bit_size}/{max_bits} ({round(bit_size/max_bits*100, 1)}%)")

    return best_word, len(best_word)


def determine_global_limits(stego_instances, rs_bytes, image_shape=(64, 64, 3)):
    """
    Computes TRUE global bit limit from actual stego_map instances (respects rep,
    block_size, etc.), then patches expected_len (and max_bits for SpreadSpectrum)
    on every instance in-place.
    """
    raw_capacities = {
        name: obj.get_capacity(image_shape)
        for name, obj in stego_instances.items()
    }
    min_bits = min(raw_capacities.values())
    fixed_bytes = min_bits // 8

    # Ensure RS doesn't eat everything
    if fixed_bytes <= rs_bytes + 2:
        rs_bytes = max(2, fixed_bytes // 4)

    max_chars = max(1, fixed_bytes - rs_bytes - len(DELIM) - 2)

    # Patch expected_len (and max_bits for SpreadSpectrum) on every instance
    for obj in stego_instances.values():
        obj.expected_len = fixed_bytes
    ss = stego_instances.get("spread_spectrum")
    if ss is not None:
        ss.max_bits = fixed_bytes * 8

    return fixed_bytes, max_chars, rs_bytes


def quick_stego_map_sanity(stego_objects, codec, image_shape=(64, 64, 3), sample_text='audit42'):
    print('\n--- STEGO MAP SANITY CHECK ---')
    print(f"Sample payload: '{sample_text}' ({len(sample_text)} chars)")
    print(f"{'Method':<20} | {'Payload Bits':<12} | {'Capacity Bits':<13} | {'Status'}")
    print('-' * 74)

    for name, obj in stego_objects.items():
        try:
            expected_len = getattr(obj, 'expected_len', fixed_byte_len)
            expected_len = fixed_byte_len if expected_len is None else int(
                expected_len)
            payload_bits = len(prepare_payload(
                sample_text, codec, expected_len=expected_len))
            capacity_bits = int(obj.get_capacity(image_shape))
            effective_cap = min(capacity_bits, expected_len * 8)
            ok = payload_bits <= effective_cap
            status = 'OK' if ok else f'OVERFLOW ({payload_bits}>{effective_cap})'
            print(f"{name:<20} | {payload_bits:<12} | {effective_cap:<13} | {status}")
        except Exception as e:
            print(f"{name:<20} | {'-':<12} | {'-':<13} | ERROR: {e}")

    print('-' * 74)

# design codex based on cornell list of eng letter freq, and put word list here to use in training


eng_freq = {
    'e': 21912, 't': 16587, 'a': 14810, 'o': 14003, 'i': 13318,
    'n': 12666, 's': 11450, 'r': 10977, 'h': 10795, 'd': 7874,
    'l': 7253, 'u': 5246, 'c': 4943, 'm': 4761, 'f': 4200, 'y': 3853,
    'w': 3819, 'g': 3693, 'p': 3316, 'b': 2715, 'v': 2019, 'k': 1257,
    'x': 315, 'q': 205, 'j': 188, 'z': 128,
}


freq_map = {}
for ch in string.ascii_lowercase:
    freq_map[ch] = eng_freq.get(ch, 1)
for ch in string.ascii_uppercase:
    freq_map[ch] = max(1, eng_freq.get(ch.lower(), 1) // 3)  # caps less common
for ch in string.digits + string.punctuation + " \0":
    freq_map[ch] = 5  # neutral weight, above rare letters

codec = HuffmanCodec.from_frequencies(freq_map)


# Build stego_map first with a placeholder expected_len (any non-None value satisfies
# LSBSteganography's guard). determine_global_limits calls get_capacity() on the real
# instances so rep, block_size, etc. are correctly accounted for, then patches
# expected_len (and max_bits for SpreadSpectrum) back onto each object in-place.

stego_map = {
    "lsb": LSBSteganography(use_header=False, expected_len=64),
    "dct": GridDCTSteganography(delta=300.0, block_size=4, rep=1, use_header=False, expected_len=64),
    "dwt": DWTSteganography(delta=250.0, band='LH', rep=1, use_header=False, expected_len=64),
    "spread_spectrum": SpreadSpectrumSteganography(gain=110.0, max_bits=64 * 8, use_header=False, expected_len=64),
    "statistical": StatisticalSteganography(block_size=4, threshold=60.0, use_header=False, expected_len=64)
}


fixed_byte_len, max_safe_word_len, RS_BYTES = determine_global_limits(
    stego_map, RS_BYTES)

if RS_BYTES >= fixed_byte_len - 2:
    RS_BYTES = fixed_byte_len // 2
    print(f"⚠️ Adjusted RS_BYTES to {RS_BYTES} to allow text room.")
rs = RSCodec(RS_BYTES)

try:
    word_list = words.words()
except:
    nltk.download('words')
    from nltk.corpus import words
    word_list = [w.lower() for w in words.words() if w.isalpha()]

max_word, max_char_len = find_max_supported_word(
    word_list, codec, fixed_byte_len)
max_char_len -= 2
safe_word_list = [w for w in word_list if len(w) <= max_char_len]


quick_stego_map_sanity(stego_map, codec)


# %%


# Check if the folder exists, then delete it
if CLEAR_BEFORE_TRAIN and os.path.exists(checkpoint_dir) and os.path.exists(models_dir):
    try:
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(models_dir)
        shutil.rmtree(data_dir)
        print(
            f"🗑️  Existing checkpoints in '{checkpoint_dir}' and models in '{models_dir}' have been cleared.")
    except Exception as e:
        print(f"⚠️  Error clearing checkpoints: {e}")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


def plot_steganography_graph(
    to_file="steganography_full_flow.pdf",
    expand_nested=False,
    dpi=300,
    show_layer_activations=False,
):
    secret_in = tf.keras.Input(shape=(64, 64, 3), name="Secret_Image_Input")
    cover_in = tf.keras.Input(shape=(64, 64, 3), name="Cover_Image_Input")

    prep_out = prep_network(secret_in)
    stego_out = hide_network([cover_in, prep_out])
    reveal_out = reveal_network(stego_out)

    viz_model = tf.keras.Model(
        inputs=[cover_in, secret_in],
        outputs=[stego_out, reveal_out],
        name="Steganography_Architecture"
    )

    return tf.keras.utils.plot_model(
        viz_model,
        to_file=os.path.join(data_dir, to_file),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=expand_nested,
        dpi=dpi,
        layer_range=None,
        show_layer_activations=show_layer_activations,
    )


class SaveEveryTen(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, max_to_keep=3):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        # Track saved epochs to know what to delete
        self.saved_epochs = []

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1

        if current_epoch % 10 == 0:
            print(f"\n✅ Milestone: Epoch {current_epoch} completed.")

            if len(self.saved_epochs) >= self.max_to_keep:
                oldest_epoch = self.saved_epochs.pop(0)
                self._delete_checkpoint(oldest_epoch)

            self._save_checkpoint(current_epoch)
            self.saved_epochs.append(current_epoch)

    def _save_checkpoint(self, epoch):
        nets = ['prep', 'hide', 'reveal']
        for net_name in nets:
            net = getattr(self.model, f"{net_name}_net")
            path = os.path.join(self.checkpoint_dir,
                                f'{net_name}_model_checkpoint_{epoch}.keras')
            net.save(path)

    def _delete_checkpoint(self, epoch):
        nets = ['prep', 'hide', 'reveal']
        for net_name in nets:
            path = os.path.join(self.checkpoint_dir,
                                f'{net_name}_model_checkpoint_{epoch}.keras')
            if os.path.exists(path):
                os.remove(path)


def save_history_and_plot(history, data_dir):
    hist_dict = history.history if hasattr(history, 'history') else history

    # Save to CSV
    filename = os.path.join(data_dir, 'training_history.csv')
    df = pd.DataFrame(hist_dict)
    df.index.name = 'epoch'
    df.to_csv(filename)

    epochs = list(range(1, len(next(iter(hist_dict.values()))) + 1))

    method_colors = {
        'none':            '#888888',
        'lsb':             '#4C78A8',
        'dct':             '#F58518',
        'dwt':             '#54A24B',
        'spread_spectrum': '#E45756',
        'statistical':     '#B279A2',
    }
    methods = list(method_colors.keys())

    # 2×2 grid:
    #   (1,1) Loss convergence   — log-Y so total/cover/secret all visible
    #   (1,2) Cover PSNR         — imperceptibility; per-method as context lines
    #   (2,1) Secret SSIM        — recovery quality; per-method as context lines
    #   (2,2) Curriculum         — LR (log primary) + noise prob (linear secondary)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Loss Convergence",
            "Cover PSNR  (Imperceptibility)",
            "Secret SSIM  (Recovery Quality)",
            "Curriculum Schedules",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}],
        ]
    )

    # ── Panel 1: Loss (log-Y) ─────────────────────────────────────────────
    # Log scale keeps total loss and the smaller cover/secret losses all
    # readable on the same axis without either collapsing to near-zero.
    for key, label, color, dash, width in [
        ('loss',        'Train Total',  '#1f77b4', 'solid', 3),
        ('val_loss',    'Val Total',    '#1f77b4', 'dash',  2),
        ('cover_loss',  'Cover',        '#d62728', 'dot',   1.5),
        ('secret_loss', 'Secret',       '#2ca02c', 'dot',   1.5),
    ]:
        if key in hist_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist_dict[key],
                name=label,
                line=dict(color=color, width=width, dash=dash),
                legendgroup='loss',
            ), row=1, col=1)

    fig.update_yaxes(title_text="Loss (log scale)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)

    # ── Panel 2: Cover PSNR ───────────────────────────────────────────────
    # Bold train/val lines as primary signal; per-method lines as dim context
    # showing how each stego method's specific cover distortion evolves.
    for key, label, dash, width in [
        ('cover_psnr',     'Train PSNR', 'solid', 3),
        ('val_cover_psnr', 'Val PSNR',   'dash',  2),
    ]:
        if key in hist_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist_dict[key],
                name=label,
                line=dict(color='#9467bd', width=width, dash=dash),
                legendgroup='psnr',
            ), row=1, col=2)

    for m in methods:
        key = f'psnr_{m}'
        if key in hist_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist_dict[key],
                name=f'PSNR/{m}',
                line=dict(color=method_colors[m], width=1, dash='dot'),
                opacity=0.4,
                legendgroup='psnr_methods',
            ), row=1, col=2)

    fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)

    # ── Panel 3: Secret SSIM ─────────────────────────────────────────────
    # Bold train/val as primary signal; per-method lines show how robustly
    # each stego embedding survives the neural pipeline.
    for key, label, dash, width in [
        ('secret_ssim',     'Train SSIM', 'solid', 3),
        ('val_secret_ssim', 'Val SSIM',   'dash',  2),
    ]:
        if key in hist_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist_dict[key],
                name=label,
                line=dict(color='#17becf', width=width, dash=dash),
                legendgroup='ssim',
            ), row=2, col=1)

    for m in methods:
        key = f'ssim_{m}'
        if key in hist_dict:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist_dict[key],
                name=f'SSIM/{m}',
                line=dict(color=method_colors[m], width=1, dash='dot'),
                opacity=0.4,
                legendgroup='ssim_methods',
            ), row=2, col=1)

    fig.update_yaxes(title_text="SSIM", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    # ── Panel 4: Curriculum — LR (log) + Noise Prob (linear) ─────────────
    # payload_len is fixed per run (determined by RS_BYTES + capacity), so it
    # would be a flat line and adds no information — omitted intentionally.
    # embed_success was removed from StegoSystem.metrics and is not logged.
    if 'lr' in hist_dict:
        fig.add_trace(go.Scatter(
            x=epochs, y=hist_dict['lr'],
            name="Learning Rate",
            line=dict(color='#e6a817', width=2),
            legendgroup='schedule',
        ), row=2, col=2, secondary_y=False)

    if 'noise_prob' in hist_dict:
        fig.add_trace(go.Scatter(
            x=epochs, y=hist_dict['noise_prob'],
            name="Noise Prob",
            line=dict(color='#27ae60', width=2),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.15)',
            legendgroup='schedule',
        ), row=2, col=2, secondary_y=True)

    fig.update_yaxes(title_text="LR (log)", type="log",
                     row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Noise Prob", range=[0, 1.05],
                     row=2, col=2, secondary_y=True)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    # ── Noise-start marker across all panels ─────────────────────────────
    if START_NOISE_EP <= epochs[-1]:
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_vline(
                x=START_NOISE_EP,
                line_dash="dash",
                line_color="rgba(100,100,100,0.35)",
                annotation_text="Noise↑" if (row == 1 and col == 1) else "",
                annotation_position="top right",
                row=row, col=col,
            )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=750, width=1400,
        title_text="Training Audit — Convergence · Imperceptibility · Recovery · Curriculum",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        # Legend to the right of the figure — never overlaps title or panels
        legend=dict(
            orientation="v",
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(r=185),
    )

    pdf_path = os.path.join(data_dir, "training_plot.pdf")
    fig.write_image(pdf_path, format="pdf", width=1400, height=750)
    fig.show()


# %%

for ablation_idx in range(len(ABLATION_CONFIGS)):
    _cfg = ABLATION_CONFIGS[ablation_idx]
    BETA = _cfg["BETA"]
    ALPHA = _cfg["ALPHA"]
    EPOCHS = _cfg["EPOCHS"]
    LEARNING_RATE = _cfg["LEARNING_RATE"]
    START_NOISE_EP = _cfg["START_NOISE_EP"]
    PEAK_NOISE_EP = _cfg["PEAK_NOISE_EP"]
    RS_BYTES = _cfg["RS_BYTES"]
    if RS_BYTES >= fixed_byte_len - 2:
        RS_BYTES = fixed_byte_len // 2
    rs = RSCodec(RS_BYTES)
    # Fresh network weights for every run — prevents weight leakage between ablations
    prep_network = build_keras_prep_network()
    hide_network = build_keras_hide_network()
    reveal_network = build_keras_reveal_network()

    # Per-run output directories
    run_ckpt_dir = os.path.join(checkpoint_dir, str(ablation_idx))
    run_models_dir = os.path.join(models_dir, str(ablation_idx))
    run_data_dir = os.path.join(data_dir, str(ablation_idx))
    for _d in (run_ckpt_dir, run_models_dir, run_data_dir):
        os.makedirs(_d, exist_ok=True)

    # Create the integrated model
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * STEPS,  # Total number of training steps
        alpha=0.01
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        global_clipnorm=1.0
    )

    # Wrap it for your StegoSystem
    tools = {
        'codec': codec,
        # Ensure keys match your 'methods' list
        **stego_map
    }

    model = StegoSystem(
        prep_net=prep_network,
        hide_net=hide_network,
        reveal_net=reveal_network,
        stego_tools=tools,
        steps_per_epoch=STEPS,
        word_list=safe_word_list,
        max_safe_chars=max_safe_word_len,
        alpha=ALPHA,
        beta=BETA,
        noise_start_epoch=START_NOISE_EP,
        noise_peak_epoch=PEAK_NOISE_EP
    )

    model.build([(None, 64, 64, 3), (None, 64, 64, 3)])
    model.compile(optimizer=optimizer, jit_compile=False)
    # Execute the plot but only on first ablation cause they won't be different logically
    if ablation_idx == 0:
        model.summary()
        plot_steganography_graph(
            "steganography_logical_flow.pdf", False, 300, False)
        plot_steganography_graph(
            "steganography_nested_flow.pdf", True, 300, False)
        plot_steganography_graph(
            "steganography_full_flow.pdf", True, 300, True)
    # Callback to save every 10 epochs
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_ckpt_dir, 'stego_best_val.weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    callbacks = [
        checkpoint_callback,
        SaveEveryTen(checkpoint_dir=run_ckpt_dir, max_to_keep=3),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=40, restore_best_weights=False
        # ),
    ]

    # Train using Keras Fit
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    clear_output(wait=True)

    model.prep_net.save(os.path.join(run_models_dir, 'prep_model.keras'))
    model.hide_net.save(os.path.join(run_models_dir, 'hide_model.keras'))
    model.reveal_net.save(os.path.join(run_models_dir, 'reveal_model.keras'))
    print("Training completed. Final model states saved.")

    results = model.evaluate(test_dataset)
    print(
        f"Validation Results - Total Loss: {results[0]:.4f}, Cover Loss: {results[1]:.4f}, Secret Loss: {results[2]:.4f}")
    save_history_and_plot(history, run_data_dir)

    del model
    tf.keras.backend.clear_session()
    gc.collect()


# %% [markdown]
# ### Final Testing / Example Output
#
# Huffman is used for all secret text, then: LSB embedding is done with [LSB_Steganography Lib](https://github.com/RobinDavid/LSB-Steganography), did ask gemini to generate basic dwt/dst to also compare with DCt/DWT embedding on images to hopefully compare robustness to the standard NN noise present in this network
#

# %%
def calculate_text_ber(original_text, revealed_text):
    """
    Calculates the Bit Error Rate between two strings.
    Handles length mismatches by penalizing missing or extra bits as errors.
    """
    # 1. Edge Cases: Total loss or empty strings
    if not original_text and not revealed_text:
        return 0.0
    if not original_text or not revealed_text:
        return 1.0  # 100% error if one is completely empty

    # 2. Convert text to bytes (UTF-8 encoding)
    # Using errors='replace' to safely handle corrupted characters
    orig_bytes = original_text.encode('utf-8', errors='replace')
    rev_bytes = revealed_text.encode('utf-8', errors='replace')

    # 3. Convert bytes to 1D numpy bit arrays
    orig_bits = np.unpackbits(np.frombuffer(orig_bytes, dtype=np.uint8))
    rev_bits = np.unpackbits(np.frombuffer(rev_bytes, dtype=np.uint8))

    # 4. Equalize lengths by padding the shorter array with zeros
    # This ensures that dropped bits or extra "garbage" bits are counted as errors
    max_len = max(len(orig_bits), len(rev_bits))

    orig_padded = np.pad(
        orig_bits, (0, max_len - len(orig_bits)), 'constant', constant_values=0)
    rev_padded = np.pad(rev_bits, (0, max_len - len(rev_bits)),
                        'constant', constant_values=0)

    # 5. Calculate mismatches
    mismatched_bits = np.sum(orig_padded != rev_padded)

    # 6. Calculate ratio (Bounded between 0.0 and 1.0)
    return mismatched_bits / max_len


def calculate_img_ber(original, revealed):

    # Ensure we are working with 8-bit integer representations
    # We use np.clip to ensure values stay within 0-255 before casting
    orig_uint8 = np.clip(original * 255, 0, 255).astype(np.uint8)
    rev_uint8 = np.clip(revealed * 255, 0, 255).astype(np.uint8)

    # Unpack the 8-bit integers into a bit array
    # axis=-1 expands the bits for each color channel
    orig_bits = np.unpackbits(orig_uint8)
    rev_bits = np.unpackbits(rev_uint8)

    # Calculate the ratio of flipped bits to total bits
    mismatched_bits = np.sum(orig_bits != rev_bits)
    total_bits = orig_bits.size

    return mismatched_bits / total_bits


def acc_txt(secret, revealed):
    if len(revealed) == 0 or len(secret) == 0:
        text_acc = 0.0
    else:
        # Compare char[i] == char[i] safely using zip
        match_count = sum(1 for s_char, r_char in zip(
            secret, revealed) if s_char == r_char)

        # Divide by the original length to get the true accuracy
        text_acc = match_count / len(secret)
    return text_acc


def check_audit_viability(text, codec, stego_objects, image_shape=(64, 64, 3), rs_overhead=RS_BYTES):
    """Accurately checks payload size vs EACH method's capacity (bit-consistent)."""

    print(f"\n--- AUDIT VIABILITY REPORT ---")
    print(f"Payload: '{text}' ({len(text)} chars)")
    print(f"{'Method':<20} | {'Max Capacity (bits)':<20} | {'Status'}")
    print("-" * 60)

    try:
        bit_array = prepare_payload(text, codec, expected_len=fixed_byte_len)
        payload_bits = len(bit_array)
    except Exception as e:
        print(f"❌ Payload preparation failed: {e}")
        return

    for name, obj in stego_objects.items():
        max_cap = int(obj.get_capacity(image_shape))  # MUST be in bits
        tool_expected_len = getattr(obj, 'expected_len', None)
        if tool_expected_len is not None:
            max_cap = min(max_cap, int(tool_expected_len) * 8)

        if payload_bits <= max_cap:
            status = "✅ OK"
        else:
            status = f"❌ OVERFLOW ({payload_bits}>{max_cap})"

        print(f"{name:<20} | {max_cap:<20} | {status}")

    print("-" * 60)


def load_weights_from_checkpoint(models_dir):
    _prep = os.path.join(models_dir, 'prep_model.keras')
    _hide = os.path.join(models_dir, 'hide_model.keras')
    _reveal = os.path.join(models_dir, 'reveal_model.keras')
    # Verify all files exist before attempting to load
    if not all(os.path.exists(p) for p in [_prep, _hide, _reveal]):
        raise FileNotFoundError(
            f"⛔ CRITICAL: One or more .keras files not found in {models_dir}. "
            "Ensure you have run the individual .save() commands first."
        )

    prep_net = tf.keras.models.load_model(_prep)
    hide_net = tf.keras.models.load_model(_hide)
    reveal_net = tf.keras.models.load_model(_reveal)
    print(f"💎 Individual weights successfully loaded from {models_dir}")
    return prep_net, hide_net, reveal_net


def to_display(img_tensor):
    # Accept BOTH tensor or numpy
    if isinstance(img_tensor, tf.Tensor):
        arr = img_tensor.numpy()
    else:
        arr = img_tensor

    # Remove batch if present
    if arr.ndim == 4:
        arr = arr[0]

    return np.clip(np.round(arr * 255), 0, 255).astype(np.uint8)


def save_visual_comparison(cover, secret, stego, reveal, method_name, index, save_dir):
    """Saves a side-by-side comparison using Plotly in PDF format."""
    os.makedirs(save_dir, exist_ok=True)

    # Create subplot layout (1 row, 3 columns)
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            "1. Original Cover", f"2. Secret ({method_name})", "3. Network Output (Stego)", "4. Reveal Image")
    )

    # Add images
    fig.add_trace(go.Image(z=cover), row=1, col=1)
    fig.add_trace(go.Image(z=secret), row=1, col=2)
    fig.add_trace(go.Image(z=stego), row=1, col=3)
    fig.add_trace(go.Image(z=reveal), row=1, col=4)

    # Update layout: Remove axes, set background to white, and adjust size
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    fig.update_layout(
        width=1200,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )

    # Save as PDF
    file_path = os.path.join(save_dir, f"sample_{index}_{method_name}.pdf")
    fig.write_image(file_path, format="pdf")


def plot_final_summary(metrics_list, save_path):
    df = pd.DataFrame(metrics_list)

    # --- 1. Data Aggregation ---
    # Calculate the average across all images for each method
    avg_df = df.groupby('method').mean().reset_index()

    # Define method order to maintain the Spatial vs Frequency logic
    spatial_methods = ['original', 'lsb', 'statistical']
    frequency_methods = ['dct', 'dwt', 'spread_spectrum']
    methods = [m for m in (spatial_methods + frequency_methods)
               if m in avg_df['method'].values]

    colors = {
        'original': '#2C3E50',       # Dark Slate
        'lsb': '#E67E22',            # Carrot Orange
        'statistical': '#F1C40F',    # Sunflower Yellow
        'dct': '#2980B9',            # Belize Blue
        'dwt': '#16A085',            # Green Sea
        'spread_spectrum': '#8E44AD'  # Wisteria Purple
    }

    # --- 2. Plot Initialization ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Avg Total Training/Audit Loss", "Avg PSNR: Cover vs Secret",
                        "Avg SSIM: Cover vs Secret", "Avg Reliability: Acc | Text BER | Img BER"),
        vertical_spacing=0.15, horizontal_spacing=0.1
    )

    # Labels for the X-axis
    labels = [m.replace('_', ' ').upper() for m in methods]

    # --- 3. Subplot Logic ---

    # Subplot 1: Total Loss
    fig.add_trace(go.Bar(
        x=labels,
        y=[avg_df[avg_df['method'] == m]['total_loss'].values[0]
            for m in methods],
        marker_color=[colors[m] for m in methods],
        name="Total Loss",
        showlegend=False
    ), row=1, col=1)

    # Subplot 2: PSNR (Grouped Bar: Cover vs Secret)
    psnr_c = [avg_df[avg_df['method'] == m]['psnr_c'].values[0]
              for m in methods]
    psnr_s = [avg_df[avg_df['method'] == m]['psnr_s'].values[0]
              for m in methods]

    fig.add_trace(go.Bar(x=labels, y=psnr_c, name='PSNR Cover',
                  marker_color='#34495E'), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=psnr_s, name='PSNR Secret',
                  marker_color='#95A5A6'), row=1, col=2)

    # Subplot 3: SSIM (Grouped Bar: Cover vs Secret)
    ssim_c = [avg_df[avg_df['method'] == m]['ssim_c'].values[0]
              for m in methods]
    ssim_s = [avg_df[avg_df['method'] == m]['ssim_s'].values[0]
              for m in methods]

    fig.add_trace(go.Bar(x=labels, y=ssim_c, name='SSIM Cover',
                  marker_color='#27AE60'), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=ssim_s, name='SSIM Secret',
                  marker_color='#2ECC71'), row=2, col=1)

    # Subplot 4: Reliability (Grouped Bar: Acc, Text BER, Img BER)
    # We filter out 'original' for reliability as it usually has no hidden data
    rel_methods = [m for m in methods if m != 'original']
    rel_labels = [m.replace('_', ' ').upper() for m in rel_methods]

    acc = [avg_df[avg_df['method'] == m]['text_acc'].values[0]
           for m in rel_methods]
    ber_t = [avg_df[avg_df['method'] == m]['ber_text'].values[0]
             for m in rel_methods]
    ber_i = [avg_df[avg_df['method'] == m]['ber_img'].values[0]
             for m in rel_methods]

    fig.add_trace(go.Bar(x=rel_labels, y=acc, name='Text Acc',
                  marker_color='#E74C3C'), row=2, col=2)
    fig.add_trace(go.Bar(x=rel_labels, y=ber_t, name='Text BER',
                  marker_color='#C0392B'), row=2, col=2)
    fig.add_trace(go.Bar(x=rel_labels, y=ber_i, name='Img BER',
                  marker_color='#F1948A'), row=2, col=2)

    # --- 4. Layout & Styling ---
    fig.update_layout(
        height=900, width=1200,
        title_text="Steganography Audit: Average Performance Comparison",
        template="plotly_white",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )

    # Axis Formatting
    fig.update_yaxes(title_text="Loss Value", row=1, col=1)
    fig.update_yaxes(title_text="dB (Higher is Better)", row=1, col=2)
    fig.update_yaxes(title_text="Index (0 to 1)", range=[0, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="Rate / Accuracy",
                     range=[0, 1.1], row=2, col=2)

    fig.write_image(save_path)
    fig.show()


# %%


def sanitize_string(s):
    """Replaces unprintable/corrupted characters with a dot (.)"""
    if not isinstance(s, str):
        return ""
    return "".join(c if c in string.printable and c not in ['\n', '\r'] else '.' for c in s)


# reload just to check saved model works, and so this can be run without fit if needed.
for ablation_idx in range(len(ABLATION_CONFIGS)):

    # Restore the RS config this ablation was trained with
    _abl_rs = ABLATION_CONFIGS[ablation_idx]["RS_BYTES"]
    if _abl_rs >= fixed_byte_len - 2:
        _abl_rs = fixed_byte_len // 2
    RS_BYTES = _abl_rs
    rs = RSCodec(RS_BYTES)

    run_models_dir = os.path.join(models_dir, str(ablation_idx))
    run_data_dir = os.path.join(data_dir, str(ablation_idx))
    os.makedirs(os.path.join(run_data_dir, 'visual_results'), exist_ok=True)
    prep_network, hide_network, reveal_network = load_weights_from_checkpoint(
        run_models_dir
    )

    all_metrics = []

    methods = ['original'] + [k.lower() for k in stego_map.keys()]

    num_samples = len(holdout_dataset)
    print(f"Evaluating {num_samples} samples...")

    sample_dataset = holdout_dataset.take(num_samples)
    pbar = tqdm(enumerate(sample_dataset), total=num_samples,
                desc="Audit Progress", unit="img")

    # Filter for medium-length words to ensure they fit your codec capacity
    max_word, max_char_len = find_max_supported_word(
        word_list, codec, fixed_byte_len)
    # subtract 2 bytes for added safety
    max_char_len -= 2
    safe_word_list = [w for w in word_list if len(w) <= max_char_len]

    random_word = random.choice(safe_word_list) + "\0"
    check_audit_viability(random_word, codec, stego_map)

    for i, (cover_tensor, secret_tensor) in pbar:
        # Prepare single sample
        secret_text = random.choice(safe_word_list) + "\0"
        sc = to_scale(to_display(cover_tensor))[tf.newaxis, ...]
        original_secret_img = to_display(secret_tensor)

        for method in methods:
            # Dynamic Tool Lookup
            tool = next((v for k, v in stego_map.items()
                        if k.lower() == method), None)

            # --- 1. Embed ---
            pre_nn_text = ""
            revealed_text = ""
            embed_success = True

            if method == 'original' or tool is None:
                new_secret = original_secret_img
            else:
                try:
                    new_secret = tool.embed(
                        original_secret_img, secret_text, codec
                    )
                    try:
                        pre_nn_text = tool.extract(new_secret, codec)
                    except:
                        pre_nn_text = "[Extract Failed]"
                except Exception as e:
                    embed_success = False
                    new_secret = original_secret_img
                    pre_nn_text = "[EMBED FAIL]"

            ss = to_scale(new_secret)[tf.newaxis, ...]

            # --- 2. Inference ---
            p_out = prep_network(ss, training=False)
            h_out = hide_network([sc, p_out], training=False)
            r_out = reveal_network(h_out, training=False)

            h_out = tf.cast(h_out, tf.float32)
            r_out = tf.cast(r_out, tf.float32)

            # --- 3. Extract & Metrics ---

            if method != 'original' and tool is not None:
                if embed_success:
                    try:
                        revealed_text = tool.extract(
                            to_display(r_out[0]), codec)
                    except:
                        revealed_text = "[Extract Failed]"
                else:
                    revealed_text = "[SKIPPED]"

                # Clean and truncate the strings for neat console formatting
                clean_sec = sanitize_string(secret_text.replace('\0', ''))
                clean_pre = sanitize_string(pre_nn_text.replace(
                    '\0', '')) if pre_nn_text else "[EMPTY]"
                clean_rev = sanitize_string(
                    revealed_text.replace('\0', ''))[:15]

                # print(
                #     f"Img {i} | {method:<15} | Sec: '{clean_sec:<12}' | PreNN: '{clean_pre:<15}' | Rev: '{clean_rev:<15}'")

            text_acc = acc_txt(secret_text, revealed_text)
            ber_text = calculate_text_ber(secret_text, revealed_text)
            ber_img = calculate_img_ber(ss, r_out)

            # print(
            #     f"text_acc: {text_acc:.2f}, ber_text: {ber_text:.2f}, ber_img: {ber_img:.2f}")
            # PSNR/SSIM
            p_c = tf.image.psnr(sc, h_out, max_val=1.0).numpy()[0]
            p_s = tf.image.psnr(ss, r_out, max_val=1.0).numpy()[0]
            s_c = tf.image.ssim(sc, h_out, max_val=1.0).numpy()[0]
            s_s = tf.image.ssim(ss, r_out, max_val=1.0).numpy()[0]

            t_loss, c_loss, s_loss = steganography_loss(sc, ss, h_out, r_out)

            # --- 4. Periodic Visual Saving ---
            # Saves every 100th image processed to avoid disk bloat
            if (i + 1) % 100 == 0:
                save_visual_comparison(
                    cover=to_display(sc[0]),
                    secret=to_display(ss[0]),
                    stego=to_display(h_out[0]),
                    reveal=to_display(r_out[0]),
                    method_name=method,
                    index=i,
                    save_dir=os.path.join(run_data_dir, 'visual_results')
                )

            # --- 5. Data Accumulation ---
            all_metrics.append({
                "image_index": i,
                "method": method,
                "total_loss": float(t_loss.numpy()),
                "psnr_c": float(p_c),
                "ssim_c": float(s_c),
                "psnr_s": float(p_s),
                "ssim_s": float(s_s),
                "ber_img": float(ber_img),
                "ber_text": float(ber_text),
                "text_acc": float(text_acc)
            })

        pbar.set_postfix({"img": i, "last_acc": f"{text_acc:.2f}"})

    # save all_metrics to csv here
    df_metrics = pd.DataFrame(all_metrics)
    final_summary_path = os.path.join(run_data_dir, 'evaluation_metrics.csv')
    df_metrics.to_csv(final_summary_path, index=False)
    print(f"Metrics saved to {final_summary_path}")

    # --- 2. Final Averaging ---
    print("\n=== Final Average Evaluation Metrics ===")
    summary = df_metrics.groupby(
        'method')[['total_loss', 'psnr_c', 'psnr_s', 'text_acc', 'ber_img', 'ber_text']].mean()
    print(summary)

    plot_final_summary(all_metrics, os.path.join(
        run_data_dir, 'final_summary_plot.pdf'))

    del prep_network, hide_network, reveal_network
    tf.keras.backend.clear_session()
    gc.collect()


# %%

def compare_ablations(ablation_configs, data_dir):
    """
    Reads training_history.csv and evaluation_metrics.csv from each
    ablation run's data directory and prints a ranked comparison table.

    Returns
    -------
    df_train  : per-run final-epoch training stats
    df_holdout: per-run per-method holdout audit stats (averaged)
    df_rank   : single-row-per-run composite ranking
    """
    train_rows = []
    holdout_rows = []

    for idx, cfg in enumerate(ablation_configs):
        run_dir = os.path.join(data_dir, str(idx))

        # — Training history (last epoch) —
        train_path = os.path.join(run_dir, 'training_history.csv')
        if os.path.exists(train_path):
            th = pd.read_csv(train_path, index_col='epoch')
            last = th.iloc[-1]
            row = {'ablation_idx': idx}
            row.update({
                'final_train_loss':    last.get('loss',           float('nan')),
                'final_val_loss':      last.get('val_loss',        float('nan')),
                'final_cover_psnr':    last.get('cover_psnr',      float('nan')),
                'final_val_psnr':      last.get('val_cover_psnr',  float('nan')),
                'final_secret_ssim':   last.get('secret_ssim',     float('nan')),
                'final_val_ssim':      last.get('val_secret_ssim', float('nan')),
                'epochs_trained':      len(th),
            })
            # Capture config fields for display
            row.update({k: v for k, v in cfg.items()})
            train_rows.append(row)
        else:
            print(
                f"[warn] No training_history.csv for ablation {idx} — skipping")

        # — Holdout evaluation metrics (mean across images, per method) —
        eval_path = os.path.join(run_dir, 'evaluation_metrics.csv')
        if os.path.exists(eval_path):
            em = pd.read_csv(eval_path)
            em['ablation_idx'] = idx
            holdout_rows.append(em)

    if not train_rows:
        print("No ablation data found — have you run the training and holdout loops yet?")
        return None, None, None

    df_train = pd.DataFrame(train_rows).set_index('ablation_idx')

    # — Holdout aggregation (exclude 'original' for text metrics) —
    if holdout_rows:
        df_hold_all = pd.concat(holdout_rows, ignore_index=True)
        # Network-level: mean over all methods including original
        net_agg = (
            df_hold_all
            .groupby('ablation_idx')[['psnr_c', 'ssim_c', 'psnr_s', 'ssim_s', 'total_loss']]
            .mean()
            .rename(columns={
                'psnr_c':     'holdout_psnr_cover',
                'ssim_c':     'holdout_ssim_cover',
                'psnr_s':     'holdout_psnr_secret',
                'ssim_s':     'holdout_ssim_secret',
                'total_loss': 'holdout_total_loss',
            })
        )
        # Text recovery: mean over non-original methods only
        text_agg = (
            df_hold_all[df_hold_all['method'] != 'original']
            .groupby('ablation_idx')[['text_acc', 'ber_text', 'ber_img']]
            .mean()
            .rename(columns={
                'text_acc':  'holdout_text_acc',
                'ber_text':  'holdout_ber_text',
                'ber_img':   'holdout_ber_img',
            })
        )
        df_holdout = net_agg.join(text_agg)
    else:
        df_holdout = pd.DataFrame()

    # — Composite ranking —
    df_rank = df_train[['BETA', 'ALPHA', 'LEARNING_RATE', 'START_NOISE_EP',
                        'RS_BYTES', 'final_val_loss', 'final_val_psnr',
                        'final_val_ssim']].copy()
    if not df_holdout.empty:
        df_rank = df_rank.join(df_holdout[['holdout_psnr_cover', 'holdout_ssim_cover',
                                           'holdout_text_acc',   'holdout_ber_text']])

    # Higher = better for PSNR/SSIM/text_acc; lower = better for loss/BER.
    # Rank each metric (1 = best) then average for a composite score.
    rank_cols = {}
    for col, ascending in [
        ('final_val_loss',      True),
        ('final_val_psnr',      False),
        ('final_val_ssim',      False),
        ('holdout_psnr_cover',  False),
        ('holdout_ssim_cover',  False),
        ('holdout_text_acc',    False),
        ('holdout_ber_text',    True),
    ]:
        if col in df_rank.columns:
            rank_cols[col] = df_rank[col].rank(
                ascending=ascending, na_option='bottom')

    if rank_cols:
        df_rank['composite_rank_score'] = pd.DataFrame(rank_cols).mean(axis=1)
        df_rank = df_rank.sort_values('composite_rank_score')

    # — Display —
    print("\n" + "=" * 72)
    print("ABLATION COMPARISON — Training (final epoch)")
    print("=" * 72)
    display(df_train[['BETA', 'ALPHA', 'LEARNING_RATE', 'RS_BYTES',
                      'START_NOISE_EP', 'epochs_trained',
                      'final_val_loss', 'final_val_psnr', 'final_val_ssim']]
            .round(4))

    if not df_holdout.empty:
        print("\n" + "=" * 72)
        print("ABLATION COMPARISON — Holdout Audit (averaged across images & methods)")
        print("=" * 72)
        display(df_holdout.round(4))

    print("\n" + "=" * 72)
    print("COMPOSITE RANKING  (lower composite_rank_score = better overall)")
    print("=" * 72)
    display(df_rank.round(4))

    best = df_rank.index[0]
    print(f"\n✅ Best overall ablation: idx {best}  — {ablation_configs[best]}")

    return df_train, df_holdout, df_rank


def plot_ablation_comparison(df_train, df_holdout, df_rank, ablation_configs, save_path=None):
    """
    2×2 Plotly figure comparing ablations across the four most diagnostic axes:
      Panel 1 (top-left)  — Cover quality:   val PSNR  + holdout PSNR cover
      Panel 2 (top-right) — Secret quality:  val SSIM  + holdout SSIM secret
      Panel 3 (bot-left)  — Text recovery:   holdout text accuracy + BER text
      Panel 4 (bot-right) — Composite rank score (lower = better)

    Each ablation is labelled with its key config params for easy identification.
    """
    if df_train is None:
        print("No ablation data to plot.")
        return

    # Build short labels: "β=5 α=1 LR=1e-3 RS=32 N@25"
    labels = []
    for i, cfg in enumerate(ablation_configs):
        lr_str = f"{cfg['LEARNING_RATE']:.0e}"
        n_str = "no-noise" if cfg['START_NOISE_EP'] >= 999 else f"N@{cfg['START_NOISE_EP']}"
        labels.append(
            f"[{i}] β={cfg['BETA']} α={cfg['ALPHA']} LR={lr_str} RS={cfg['RS_BYTES']} {n_str}")

    idx_list = list(range(len(ablation_configs)))

    # Colour palette — one colour per ablation, consistent across panels
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    colors = [palette[i % len(palette)] for i in idx_list]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cover Quality  (PSNR — higher is better)",
            "Secret Recovery Quality  (SSIM — higher is better)",
            "Text Recovery  (Accuracy ↑  |  BER ↓)",
            "Composite Rank Score  (lower = better overall)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True},  {"secondary_y": False}],
        ],
    )

    # ── Panel 1: PSNR (cover) ─────────────────────────────────────────────
    # Val PSNR from training history
    if 'final_val_psnr' in df_train.columns:
        fig.add_trace(go.Bar(
            x=labels,
            y=[df_train.loc[i, 'final_val_psnr'] if i in df_train.index else float('nan')
               for i in idx_list],
            name="Val PSNR (train)",
            marker_color=colors,
            opacity=0.6,
            showlegend=True,
        ), row=1, col=1)

    # Holdout PSNR cover
    if df_holdout is not None and 'holdout_psnr_cover' in df_holdout.columns:
        fig.add_trace(go.Scatter(
            x=labels,
            y=[df_holdout.loc[i, 'holdout_psnr_cover'] if i in df_holdout.index else float('nan')
               for i in idx_list],
            name="Holdout PSNR cover",
            mode='markers+lines',
            marker=dict(size=10, symbol='diamond', color=colors,
                        line=dict(width=1, color='black')),
            line=dict(color='black', width=1, dash='dot'),
        ), row=1, col=1)

    # ── Panel 2: SSIM (secret) ────────────────────────────────────────────
    if 'final_val_ssim' in df_train.columns:
        fig.add_trace(go.Bar(
            x=labels,
            y=[df_train.loc[i, 'final_val_ssim'] if i in df_train.index else float('nan')
               for i in idx_list],
            name="Val SSIM (train)",
            marker_color=colors,
            opacity=0.6,
            showlegend=True,
        ), row=1, col=2)

    if df_holdout is not None and 'holdout_ssim_secret' in df_holdout.columns:
        fig.add_trace(go.Scatter(
            x=labels,
            y=[df_holdout.loc[i, 'holdout_ssim_secret'] if i in df_holdout.index else float('nan')
               for i in idx_list],
            name="Holdout SSIM secret",
            mode='markers+lines',
            marker=dict(size=10, symbol='diamond', color=colors,
                        line=dict(width=1, color='black')),
            line=dict(color='black', width=1, dash='dot'),
        ), row=1, col=2)

    # ── Panel 3: Text accuracy (primary) + BER text (secondary) ──────────
    if df_holdout is not None and 'holdout_text_acc' in df_holdout.columns:
        fig.add_trace(go.Bar(
            x=labels,
            y=[df_holdout.loc[i, 'holdout_text_acc'] if i in df_holdout.index else float('nan')
               for i in idx_list],
            name="Text Accuracy",
            marker_color=colors,
            opacity=0.75,
        ), row=2, col=1, secondary_y=False)

    if df_holdout is not None and 'holdout_ber_text' in df_holdout.columns:
        fig.add_trace(go.Scatter(
            x=labels,
            y=[df_holdout.loc[i, 'holdout_ber_text'] if i in df_holdout.index else float('nan')
               for i in idx_list],
            name="BER Text (lower = better)",
            mode='markers+lines',
            marker=dict(size=9, color=colors,
                        line=dict(width=1, color='black')),
            line=dict(color='crimson', width=2, dash='dash'),
        ), row=2, col=1, secondary_y=True)

    # ── Panel 4: Composite rank score ─────────────────────────────────────
    if df_rank is not None and 'composite_rank_score' in df_rank.columns:
        scores = [df_rank.loc[i, 'composite_rank_score'] if i in df_rank.index else float('nan')
                  for i in idx_list]
        fig.add_trace(go.Bar(
            x=labels,
            y=scores,
            name="Composite rank score",
            marker_color=colors,
            opacity=0.85,
            showlegend=False,
        ), row=2, col=2)
        # Annotate the winner
        best_i = int(df_rank.index[0])
        if best_i < len(labels):
            fig.add_annotation(
                x=labels[best_i], y=scores[best_i],
                text="★ Best",
                showarrow=True, arrowhead=2, yshift=12,
                font=dict(color='green', size=12),
                row=2, col=2,
            )

    # ── Axis labels ───────────────────────────────────────────────────────
    fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)
    fig.update_yaxes(title_text="SSIM (0–1)", range=[0, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="Text Accuracy (0–1)", range=[0, 1.05],
                     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="BER Text (0–1)", range=[0, 1.05],
                     row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Avg Rank (lower = better)", row=2, col=2)

    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(tickangle=-35, tickfont=dict(size=9),
                         row=row, col=col)

    fig.update_layout(
        height=800, width=1600,
        title_text="Ablation Study: Cover Quality · Secret Recovery · Text Accuracy · Composite Rank",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )

    if save_path:
        fig.write_image(save_path, format="pdf", width=1600, height=800)
        print(f"Ablation comparison plot saved to {save_path}")
    fig.show()


df_abl_train, df_abl_holdout, df_abl_rank = compare_ablations(
    ABLATION_CONFIGS, data_dir)

df_abl_rank.to_csv(os.path.join(data_dir, "ablation_comparison_rank.csv"))

if df_abl_rank is not None:
    plot_ablation_comparison(
        df_abl_train, df_abl_holdout, df_abl_rank,
        ABLATION_CONFIGS,
        save_path=os.path.join(data_dir, "ablation_comparison.pdf"),
    )


# %% [markdown]
# ### Final testing for steg methods
#
# once best ablation has been determined, should speed up dev to then use below on that one, instead of needing to run on each...
#

# %%
df_abl_rank.to_csv(os.path.join(data_dir, "ablation_comparison_rank.csv"))


# %%

# =============================================================================
# DCT & DWT Parameter Sweep — repetition (rep) configs
#
# rep=1  → original behaviour (no redundancy)
# rep=2  → each payload bit embedded in 2 consecutive slots, majority voted
# rep=3  → each payload bit embedded in 3 consecutive slots, majority voted
#
# delta / band / block_size are kept at the previous best values so that
# the only variable changing is the redundancy factor.
#
# Capacity constraint: rep reduces effective capacity by factor rep,
# so rep=3 with block_size=4 on a 64×64 image gives 768/3 = 256 effective bits.
# fixed_byte_len * 8 must be <= that; the embed will raise if not.
# =============================================================================


def evaluate_grid_with_holdout(method_name, param_grid, codec, n_samples=None, dataset=None,
                               safe_words=None, verbose=False,
                               prep_net=None, hide_net=None, reveal_net=None):
    """
    Evaluate parameter grid using holdout images and a specific set of network weights.
    Runs the full pipeline: stego tool embed -> prep/hide/reveal networks -> stego tool extract on revealed image.
    Returns a pandas DataFrame with param columns + `ber_text` and `text_acc`.

    prep_net / hide_net / reveal_net: explicit network instances to use. Falls back to
    the module-level globals (prep_network, hide_network, reveal_network) if not provided.
    """

    dataset = sample_dataset if dataset is None else dataset
    safe_words = safe_word_list if safe_words is None else safe_words
    _prep = prep_net if prep_net is not None else prep_network
    _hide = hide_net if hide_net is not None else hide_network
    _reveal = reveal_net if reveal_net is not None else reveal_network

    # collect up to n_samples pairs from the tf.data.Dataset-like object
    pairs = []
    for i, item in enumerate(dataset):
        if n_samples is not None and i >= n_samples:
            break
        cover_tensor, secret_tensor = item
        cover_img = to_display(cover_tensor)
        secret_img = to_display(secret_tensor)
        pairs.append((cover_img, secret_img))

    if len(pairs) == 0:
        raise ValueError('No samples found in sample_dataset')

    results = []
    for params in param_grid:
        try:
            if method_name.lower() == 'dct':
                stg = GridDCTSteganography(delta=params.get('delta', 64.0),
                                           block_size=params.get(
                                               'block_size', 4),
                                           use_header=params.get(
                                               'use_header', False),
                                           expected_len=fixed_byte_len,
                                           rep=params.get('rep', 1))
            elif method_name.lower() == 'dwt':
                stg = DWTSteganography(delta=params.get('delta', 110.0),
                                       band=params.get('band', 'LH'),
                                       use_header=params.get(
                                           'use_header', False),
                                       expected_len=fixed_byte_len,
                                       rep=params.get('rep', 1))
            elif method_name.lower() == 'spread_spectrum':
                stg = SpreadSpectrumSteganography(
                    gain=params.get('gain', 100.0),
                    max_bits=fixed_byte_len * 8,
                    use_header=params.get('use_header', False),
                    expected_len=fixed_byte_len)
            elif method_name.lower() == 'statistical':
                stg = StatisticalSteganography(
                    block_size=params.get('block_size', 4),
                    threshold=params.get('threshold', 60.0),
                    use_header=params.get('use_header', False),
                    expected_len=fixed_byte_len)
            else:
                raise NotImplementedError(
                    f"Supported methods: 'dct', 'dwt', 'spread_spectrum', 'statistical'; got {method_name}")
        except Exception as e:
            row = dict(params)
            row.update(
                {'ber_text': np.nan, 'text_acc': np.nan, 'error': str(e)})
            results.append(row)
            if verbose:
                print('Instantiation error for', params, e)
            continue

        bers = []
        accs = []
        for cover_img, secret_img in pairs:
            # pick a safe short word and append terminator via chr(0) to avoid JSON escape issues
            secret_text = random.choice(safe_words) + chr(0)
            try:
                new_secret = stg.embed(secret_img, secret_text, codec)
            except Exception as e:
                new_secret = secret_img
                if verbose:
                    print('Embed error', e)

            # Run through the neural pipeline (prep -> hide -> reveal)
            sc = to_scale(to_display(cover_img))[tf.newaxis, ...]
            ss = to_scale(new_secret)[tf.newaxis, ...]
            p_out = _prep(ss, training=False)
            h_out = _hide([sc, p_out], training=False)
            r_out = _reveal(h_out, training=False)

            try:
                revealed_text = stg.extract(to_display(r_out[0]), codec)
            except Exception:
                revealed_text = '[Extract Failed]'

            bers.append(calculate_text_ber(secret_text, revealed_text))
            accs.append(acc_txt(secret_text, revealed_text))

        row = dict(params)
        row.update({'ber_text': float(np.mean(bers)),
                   'text_acc': float(np.mean(accs))})
        results.append(row)

    return pd.DataFrame(results)


# %%

# =============================================================================
# Stego Parameter Sweep — per-ablation
#
# Iterates over every trained ablation, loads its weights, then runs a full
# DCT / DWT / Spread-Spectrum / Statistical parameter sweep against those
# specific network weights. This lets us separate the effect of the stego
# params from the effect of the network training config.
#
# Config rationale (informed by round-1 sweep results):
#   DCT : delta<128 all failed in round 1; delta=200–400 confirmed working.
#          rep=3 adds majority-vote redundancy at the cost of 3× capacity.
#   DWT : LL band survived neural smoothing better than LH in round 1;
#          delta=200–400 needed. LH retained at delta=300 as comparison point.
#   SS  : Robustness-trained models (idx 2,4,5,6 — early noise) apply stronger
#          per-step smoothing; gain=200–600 range extended to compensate.
#   Stat: threshold=80–200 was productive in round 1; threshold=300 added for
#          robustness-trained models where moderate thresholds may not survive.
# =============================================================================

# --- Stego param grids (shared across all ablations) ---
_dct_configs = [
    # Config 1 — dct_d150_b4: lower-bound probe for #4-variants (idx 9/10)
    # β↓ and α↑ in idx 9/10 should relax smoothing slightly. If d150 works here
    # it means reduced β lets the network pass finer block perturbations, giving
    # better pre-network PSNR than d200 at the same text_acc.
    {'delta': 150.0, 'block_size': 4, 'rep': 1, 'label': 'dct_d150_b4_rep1'},

    # Config 2 — dct_d200_b4: carry-forward known winner for #4 (0.6 text_acc)
    # Retested here as a direct benchmark for the new #4-variants vs the original.
    {'delta': 200.0, 'block_size': 4, 'rep': 1, 'label': 'dct_d200_b4_rep1'},

    # Config 3 — dct_d200_b8: same delta as the #4 winner but larger block
    # block_size=8 modifies 4× fewer spatial blocks than block_size=4, so the
    # cover PSNR overhead per-image is significantly lower. The tradeoff is that
    # each block carries proportionally more signal; if the network averages within
    # blocks and the block is larger, individual QIM quantisation steps survive
    # better because there are fewer discontinuities at block boundaries.
    {'delta': 200.0, 'block_size': 8, 'rep': 1, 'label': 'dct_d200_b8_rep1'},

    # Config 4 — dct_d300_b4: bridge delta for #3-variants (idx 7/8)
    # #3 failed at d200 but earlier noise (idx 7) should reduce smoothing.
    # d300 is the first step above the #4 range — if idx 7/8 still smooth d200,
    # d300 has a wider quantisation margin to survive.
    {'delta': 300.0, 'block_size': 4, 'rep': 1, 'label': 'dct_d300_b4_rep1'},

    # Config 5 — dct_d300_b8: high delta + large block for #3-variants
    # Combining d300 amplitude with the reduced spatial footprint of block_size=8
    # targets both goals simultaneously for #3-variant networks: the large margin
    # survives heavier smoothing (cover PSNR already high → can afford larger δ),
    # and the reduced block count means the cover image has fewer modified regions.
    {'delta': 300.0, 'block_size': 8, 'rep': 1, 'label': 'dct_d300_b8_rep1'},

    # Config 6 — dct_d500_b4: extreme delta for #3-baseline and its variants
    # #3's original sweep had d200–d400 all fail completely; the step to d500
    # moves well outside the network's linear adjustment range. At this amplitude
    # the QIM shift after IDCT reconstruction is detectable even after the neural
    # smoothing pass. If d500 still fails for #3, it confirms the network is doing
    # structural suppression, not just amplitude damping.
    {'delta': 500.0, 'block_size': 4, 'rep': 1, 'label': 'dct_d500_b4_rep1'},
]

_dwt_configs = [
    # Config 1 — dwt_LH_d250: fine-tune below #4's LH d300 baseline (0.4 text_acc)
    # For idx 9/10 where β is reduced, the network's smoothing is slightly relaxed.
    # d250 probes whether that marginal relaxation allows a lower-distortion delta
    # to pass, improving cover PSNR from the stego image's perspective.
    {'delta': 250.0, 'band': 'LH', 'rep': 1, 'label': 'dwt_LH_d250_rep1'},

    # Config 2 — dwt_LH_d300: carry-forward #4 baseline (0.4 text_acc)
    # Direct comparison point for all new ablations.
    {'delta': 300.0, 'band': 'LH', 'rep': 1, 'label': 'dwt_LH_d300_rep1'},

    # Config 3 — dwt_LH_d400: extend upward from #4 baseline
    # If idx 7/8's earlier noise preserved frequency sensitivity, d400 on LH
    # should be the next viable step up. For #4-variants, d400 may also work
    # if the slightly reduced β makes the network slightly less aggressive.
    {'delta': 400.0, 'band': 'LH', 'rep': 1, 'label': 'dwt_LH_d400_rep1'},

    # Config 4 — dwt_HL_d300: untested vertical-edge band
    # The current DWT uses Haar decomposition on (H, W). LH captures horizontal
    # frequency variations (vertical edges in the image); HL captures vertical
    # frequency variations (horizontal edges). The residual convolutional kernels
    # in the reveal network are isotropic (3×3, 4×4, 5×5), but the hide network's
    # smoothing effect may be directionally asymmetric in practice — HL may survive
    # better or worse than LH depending on dataset texture distribution (TinyImageNet
    # has mixed natural + synthetic scenes). This is a low-cost hypothesis test.
    {'delta': 300.0, 'band': 'HL', 'rep': 1, 'label': 'dwt_HL_d300_rep1'},

    # Config 5 — dwt_LH_d500: high delta for #3 variants (idx 7/8)
    # #3's original sweep failed across all tested DWT configs (max d400).
    # idx 7/8 start noise 15 epochs earlier, so the network should suppress
    # frequency signals slightly less aggressively. d500 provides a large enough
    # QIM margin to survive the residual smoothing that earlier-noise training
    # does not fully eliminate.
    {'delta': 500.0, 'band': 'LH', 'rep': 1, 'label': 'dwt_LH_d500_rep1'},
]

_ss_configs = [
    # Sub-floor probes — densely cover the gap below the confirmed minimum (g150).
    # g50 is an extreme lower bound; if it passes, the method is extremely robust.
    # g70/g90 are the primary interest range: large enough to survive neural
    # smoothing but small enough to meaningfully reduce cover distortion vs g150.
    # g110/g130 are safety fallbacks in case the floor is closer to 150 than expected.
    {'gain': 50.0,  'label': 'ss_g50'},
    {'gain': 70.0,  'label': 'ss_g70'},
    {'gain': 90.0,  'label': 'ss_g90'},
    {'gain': 110.0, 'label': 'ss_g110'},
    {'gain': 130.0, 'label': 'ss_g130'},
]

_stat_configs = [
    # Sub-floor probes — densely cover the gap below the confirmed minimum (t80).
    # t20 is an aggressive lower bound to characterise the failure mode.
    # t35/t50 are the primary interest range.
    # t60/t70 are safety fallbacks bridging to the confirmed-working t80.
    # All use block_size=4 — block_size=8 always fails due to capacity overflow
    # (64 bits available vs 256 bits required), not network robustness.
    {'block_size': 4, 'threshold': 20.0, 'label': 'stat_b4_t20'},
    {'block_size': 4, 'threshold': 35.0, 'label': 'stat_b4_t35'},
    {'block_size': 4, 'threshold': 50.0, 'label': 'stat_b4_t50'},
    {'block_size': 4, 'threshold': 60.0, 'label': 'stat_b4_t60'},
    {'block_size': 4, 'threshold': 70.0, 'label': 'stat_b4_t70'},
]


def run_stego_param_sweep_per_ablation(ablation_configs, data_dir, models_dir,
                                       codec, holdout_dataset, safe_word_list,
                                       n_samples=5, verbose=False, **kwargs):
    """
    For every ablation, loads its saved weights and runs the full stego
    parameter sweep (DCT, DWT, Spread-Spectrum, Statistical).

    Saves per-ablation CSVs to data/<idx>/param_sweep_<method>.csv and
    a cross-ablation combined ranking to data/stego_sweep_combined.csv.

    Returns
    -------
    df_combined : DataFrame with all results, columns including
                  ablation_idx, method, label, text_acc, ber_text.
    df_best     : Best config per method per ablation (top text_acc row).
    """
    all_rows = []

    target_ablations = kwargs.get(
        'target_ablations', range(len(ablation_configs)))
    for ablation_idx in target_ablations:
        cfg = ablation_configs[ablation_idx]
        print("\n" + "=" * 68)
        print(f"STEGO SWEEP — ablation {ablation_idx}  "
              f"(β={cfg['BETA']}, α={cfg['ALPHA']}, "
              f"LR={cfg['LEARNING_RATE']}, noise_start={cfg['START_NOISE_EP']}, "
              f"RS={cfg['RS_BYTES']})")
        print("=" * 68)

        run_models_dir = os.path.join(models_dir, str(ablation_idx))
        run_data_dir = os.path.join(data_dir, str(ablation_idx))
        os.makedirs(run_data_dir, exist_ok=True)

        try:
            _prep, _hide, _reveal = load_weights_from_checkpoint(
                run_models_dir)
        except Exception as e:
            print(
                f"  [skip] Could not load weights for ablation {ablation_idx}: {e}")
            continue

        method_frames = []
        for method_name, grid in [
            ('dct',              _dct_configs),
            ('dwt',              _dwt_configs),
            ('spread_spectrum',  _ss_configs),
            ('statistical',      _stat_configs),
        ]:
            print(
                f"  Running {method_name} sweep ({len(grid)} configs × {n_samples} samples)...")
            df = evaluate_grid_with_holdout(
                method_name, grid, codec,
                n_samples=n_samples,
                dataset=holdout_dataset,
                safe_words=safe_word_list,
                verbose=verbose,
                prep_net=_prep,
                hide_net=_hide,
                reveal_net=_reveal,
            )
            df['method'] = method_name
            df['ablation_idx'] = ablation_idx
            df_sorted = df.sort_values(
                ['text_acc', 'ber_text'], ascending=[False, True])
            df_sorted.to_csv(
                os.path.join(run_data_dir, f'param_sweep_{method_name}.csv'),
                index=False)
            method_frames.append(df_sorted)

        if not method_frames:
            continue

        df_abl = pd.concat(method_frames, ignore_index=True)
        df_abl['ablation_idx'] = ablation_idx

        del _prep, _hide, _reveal, method_frames
        tf.keras.backend.clear_session()
        gc.collect()

        # Per-ablation combined summary
        print(f"\n  TOP-5 configs for ablation {ablation_idx} (all methods):")
        display(
            df_abl.sort_values(['text_acc', 'ber_text'],
                               ascending=[False, True])
            [['method', 'label', 'text_acc', 'ber_text']].head(5)
        )

        all_rows.append(df_abl)

    if not all_rows:
        print("No sweep results — have the ablation models been trained yet?")
        return None, None

    df_combined = pd.concat(all_rows, ignore_index=True)

    # Best config per (method, ablation) — highest text_acc, lowest ber_text
    df_best = (
        df_combined
        .sort_values(['text_acc', 'ber_text'], ascending=[False, True])
        .groupby(['ablation_idx', 'method'], sort=False)
        .first()
        .reset_index()
    )

    # Cross-ablation ranking: for each config label, mean text_acc across ablations
    df_cross = (
        df_combined
        .groupby(['method', 'label'])[['text_acc', 'ber_text']]
        .mean()
        .rename(columns={'text_acc': 'mean_text_acc', 'ber_text': 'mean_ber_text'})
        .sort_values(['mean_text_acc', 'mean_ber_text'], ascending=[False, True])
        .reset_index()
    )

    print("\n" + "=" * 68)
    print("CROSS-ABLATION STEGO CONFIG RANKING  (mean across all ablations)")
    print("=" * 68)
    display(df_cross.head(10))

    # Save combined outputs
    df_combined.to_csv(os.path.join(
        data_dir, 'stego_sweep_combined.csv'), index=False)
    df_best.to_csv(os.path.join(
        data_dir, 'stego_sweep_best_per_ablation.csv'), index=False)
    df_cross.to_csv(os.path.join(
        data_dir, 'stego_sweep_cross_ablation_ranking.csv'), index=False)
    print(f"\nSaved: stego_sweep_combined.csv, stego_sweep_best_per_ablation.csv, "
          f"stego_sweep_cross_ablation_ranking.csv → {data_dir}")

    return df_combined, df_best


df_sweep_combined, df_sweep_best = run_stego_param_sweep_per_ablation(
    ABLATION_CONFIGS, data_dir, models_dir,
    codec, holdout_dataset, safe_word_list, n_samples=30, verbose=False,
)

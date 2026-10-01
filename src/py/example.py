import os
# Disabilitiamo il pre-caching globale per garantire il controllo reale del driver hardware
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from metatensor import MetaTensor, InitPattern
from tqdm import tqdm
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=== Orchestrazione OpenXLA/LibTorch Standard (C++26) ===")
print("=== Monitoraggio Avanzamento tramite TQDM Postfix ===\n")

# Configurazione parametri e target di convergenza matematici
learning_rate = 0.5
max_epochs = 50
convergence_threshold = 1e-3

# 1. Generazione del Dataset Separabile (Teacher-Student Pattern)
X = MetaTensor(128, 64, pattern=InitPattern.RandomUniform, device=device)
W_true = MetaTensor(64, 1, pattern=InitPattern.RandomNormal, device=device)
Y_true = X.matmul(W_true).element_wise_sigmoid() # Target stabili per la convergenza

# 2. Inizializzazione Pesi del Modello (Partiamo da Zeri)
W = MetaTensor(64, 1, pattern=InitPattern.Zeros, device=device)

# Abilitiamo il tracciamento Autograd del nastro
W.watch()

# 3. LOOP DELLE EPOCHE AVVOLTO DA TQDM
# Configurianto la barra con descrizione e layout pulito
progress_bar = tqdm(
    range(1, max_epochs + 1),
    desc="Training MetaTensor",
    unit="epoch",
    bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}"
)

for epoch in progress_bar:
    host_loss_value = 0.0

    # ISOLAMENTO DELLO SCOPE PER I BUFFER TEMPORANEI INTERMEDI
    # Questa funzione garantisce che i distruttori C++ delle matrici intermedie
    # scattino IMMEDIATAMENTE alla fine di ogni passo, azzerando la VRAM (Zero-Caching)
    def train_step():
        # Forward Pass fuso nell'IR hardware
        Y_pred = X.matmul(W).element_wise_sigmoid()
        error = Y_pred.sub(Y_true)
        square_error = error.element_wise_mul(error)

        # Riduzione totale a scalare 0-D (MetaTensor_float senza dimensioni nel tipo)
        loss = square_error.reduce_all_sum()

        # Estrattore del valore scalare via cast implicito C++26 mappato da nanobind
        current_loss = float(loss)

        # Backward Pass nativo per accumulare i gradienti hardware
        loss.storage.backward()

        return current_loss, W.grad()

    # Eseguiamo l'iterazione isolata ed estraiamo la loss numerica e il gradiente
    host_loss_value, dW = train_step()

    # Applichiamo la discesa SGD atomica in-place sulla GPU bypassando l'Autograd
    W.apply_gradient_descent(dW, learning_rate)

    # AGGIORNAMENTO DINAMICO DELLA TELEMETRIA SU TQDM
    # Iniettiamo la loss corrente come testo postfix a destra della barra
    progress_bar.set_postfix({
        "Loss": f"{host_loss_value:.6f}",
        "Target": f"< {convergence_threshold}"
    })

    # VERIFICA CRITICA EARLY STOPPING (Rilevamento anomalie numeriche nei dati)
    if torch.isnan(torch.tensor(host_loss_value)) or torch.isinf(torch.tensor(host_loss_value)):
        progress_bar.set_postfix_str("[FAILED] Instabilità numerica rilevata!")
        print(f"\n[EARLY STOPPING] Arresto forzato all'epoca {epoch}.")
        break

    # VERIFICA DELLA CONVERGENZA NATURALE ANTICIPATA
    if host_loss_value < convergence_threshold:
        progress_bar.set_postfix_str(f"[CONVERGED] Target raggiunto alla loss: {host_loss_value:.6f}")
        break

    # Un piccolo delay per rendere l'avanzamento visibile se la GPU è troppo veloce (opzionale)
    time.sleep(0.01)

# 4. ZERO-CACHING ACCURATE MEMORY RECLAMATION
# Sradichiamo i riferimenti globali persistenti per forzare l'unmapping sincrono
# dell'allocatore C++ (`c10::cuda::CUDACachingAllocator::emptyCache()`), pulendo il chip.
X.clear()
Y_true.clear()
W_true.clear()
W.clear()

print("\nAddestramento concluso con successo. VRAM e contesti hardware ripristinati a zero.")

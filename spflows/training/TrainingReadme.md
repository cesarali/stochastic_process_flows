# Setting MLFlow in Uni Potsdam Cluster

### Steps:

1. **Open an interactive session:**
   ```bash
   srun -c 2 --time 4:00:00 --pty bash
   ```

2. **Navigate to the directory where `mlruns` is located (results folder), activate the environment, and open MLFlow server with IP awareness:**
   ```bash
   mlflow server --host $(hostname -I | cut -f1 -d' ') --port 5000
   ```

3. **On your local computer, open an SSH tunnel (remember to change the node and username):**
   ```bash
   ssh -N -L 5000:n-hpc-caX:5000 username@login1.hpc.uni-potsdam.de
   ```

4. **Access MLFlow in your local browser:**
   Open the following URL:
   ```
   http://localhost:5000/
   ```

---

### Additional Steps:

5. **Set an interactive session:**
   ```bash
   srun --cpus-per-task=4 -p gpu --gpus=tesla_v100:1 --pty bash
   ```

6. **Copy files to the cluster:**
   ```bash
   scp 2024-12-18.tar ojedamarin@login1.hpc.uni-potsdam.de:~/Projects/Portfolio/stochastic_process_flows/data/raw/gecko
   ```

7. **Call the trainer:**
   Navigate to the location `scripts\training` and run:
   ```bash
   /home/ojedamarin/.conda/envs/torchts/bin/python train_gecko.py --epochs 10 --num_batches_per_epoch 50 --residual_layers 8
   ```

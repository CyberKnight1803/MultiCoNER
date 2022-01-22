# MultiCoNER Model
A RoBERTa-BiLSTM-CRF based approach for Named Entity Recognition.

### Framework and technologies -
* CoNLL DataModule for prepeocessing
* HuggingFace pre-trained transformer models
* PyTorch and PyTorch-Lightning for training and testing 

#### Arguments - 
```
    parser.add_argument('--lr', type=float, default=3e-5, help="learning_rate")
    parser.add_argument('--epochs', type=int, default=5, help="epochs")
    parser.add_argument('--base_model', type=str, default="xlm-roberta-base", help="Base model")
    parser.add_argument('--seq_length', type=int, default=32, help="max_seq_length")
    parser.add_argument('--padding', type=str, default="max_length", help="paddind type")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--tagging_scheme', type=str, default='wnut', help="Tagging scheme")
    parser.add_argument('--gpu', type=int, default=1, help="GPUS")
    parser.add_argument('--workers', type=int, default=4, help="CPU threads")
    parser.add_argument('--nodes', type=int, default=1, help="num nodes")
    parser.add_argument('--strategy', type=str, default="ddp", help="cluster training strategy")
    parser.add_argument('--dataset', type=str, default=None, required=True, help="dataset path")
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--name', type=str, default="trails", help="run names")
    parser.add_argument('--checkpoint', type=bool, default=False, help="Checkpointing")
```

#### Running
To run on the HPC use the `job.sh` slurm script to schedule using the following command - `sbatch job.sh`


#### Request for gpu, cpu and memory

```
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --job-name="modx"
#SBATCH -o outputs/slurm.%j.out
#SBATCH -e outputs/slurm.%j.err
#SBATCH --mail-user=email_address
#SBATCH --mail-type=END
#SBATCH --export=ALL
```


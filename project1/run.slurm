#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=cs_dept                          # need to select 'gpu' QoS
#SBATCH --job-name=graphrag
#SBATCH --output=/scratch/dmeher/slurm_outputs/graphrag.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/graphrag.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem-per-cpu=80GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00                   # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=dmeher@gmu.edu   # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

#source /scratch/dmeher/custom_env/recguru_env/bin/activate
module load ollama
#module load gnu10
#module load python
# Load Miniforge (only necessary if Conda is not initialized globally)
source /scratch/dmeher/custom_env/miniforge/etc/profile.d/conda.sh
conda activate graphrag_env

# Check if the Ollama server is running on the expected node
echo "Running on node: $SLURMD_NODENAME"
curl http://localhost:11434/v1/models  # This is to check if the Ollama server is up

curl -X POST http://localhost:11434/v1/embeddings -H "Content-Type: application/json" -d '{"model": "nomic-embed-text", "input": "Test embedding generation with nomic model"}'

/scratch/dmeher/custom_env/miniforge/envs/graphrag_env/bin/python -m graphrag.query --root ./ragtest         --method local         "Who are the main demons Krishna defeated during his childhood?"

/scratch/dmeher/custom_env/miniforge/envs/graphrag_env/bin/python -m graphrag.query         --root ./ragtest         --method global         "How does Krishna represent different aspects of divinity in Hinduism?"

/scratch/dmeher/custom_env/miniforge/envs/graphrag_env/bin/python -m graphrag.query \
	--root ./ragtest \
	--method global \
	"What are the top themes in this story?"
	       
/scratch/dmeher/custom_env/miniforge/envs/graphrag_env/bin/python -m graphrag.index --root ./ragtest
#python -m graphrag.index --root ./ragtest
#curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./ragtest/input/book.txt

vim /scratch/dmeher/custom_env/miniforge/envs/graphrag_env/lib/python3.10/site-packages/graphrag/llm/openai/openai_embeddings_llm.py

salloc -p contrib-gpuq -q cs_dept --ntasks-per-node=2 --gres=gpu:3g.40gb:1 -t 0-2:00:00 --mem=12GB
module load ollama
ollama serve
Note the name of the node you receive, (gpu012, gpu018, etc). Then in a second terminal run:

ssh gpu###
module load ollama
ollama run gemma2


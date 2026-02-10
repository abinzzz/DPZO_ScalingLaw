More on enviroments can be found in `environments.yml`. You can also create one using commands below.
```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero
```


CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-125m TASK=SQuAD MODE=ft LR=1e-5 EPS=1e-3 EVAL_STEPS=125 DP_SAMPLE_RATE=0.032 DP_EPS=2.0 STEPS=500 N=4 DP_CLIP=7.5 bash examples/dpaggzo.sh


CUDA_VISIBLE_DEVICES=1,2 MODEL=facebook/opt-125m TASK=SQuAD MODE=ft LR=1e-5 EPS=1e-3 EVAL_STEPS=125 DP_SAMPLE_RATE=0.032 DP_EPS=2.0 STEPS=500 N=4 DP_CLIP=7.5 NGPU=2 bash examples/dpaggzo.sh

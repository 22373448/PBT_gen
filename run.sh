export OPENAI_API_KEY="sk-4b4837e283714bc39d8b6546eb483718"
export TRANSFORMER_PATH='/mnt/c/Users/34203/Desktop/test-generation/my_transformer'


python -m pbt_gen.cli --project-dir /mnt/c/Users/34203/Desktop/PBT_gen/example_project \
        --output-dir /mnt/c/Users/34203/Desktop/PBT_gen/example_project/tests \
        --functions pkg.module_a.Encoder.encode \
        --top-n 10 \
        --python-path /home/ler/miniconda3/envs/test4dt/bin/python
        
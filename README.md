We present the process of fine-tuning using the [LoRA](https://github.com/andysdc/LLaMA-Efficient-Tuning) method. 

## Data Preparation
You need to download the Spider dataset to the folder `./dataset/spider`.

## Run

### Origin Instruction Data Generation
```
python src/data_convert_to_sft.py
```
### Error Identification
Based on the [evaluation](https://github.com/taoyds/test-suite-sql-eval), we identify system errors and value errors.:
```
bash src/evaluation_system_values.sh
```
Skeleton Error:
```
python src/skeleton_extra.py
python src/cl_gobal.py
python src/cos.py
```


### Error Correction
```
python src/error_data_prompt.py
python src/error_data_prompt.py
```

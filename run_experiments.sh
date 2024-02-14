data=p3
for seed in 42 43 44
do
for size in small base large xl
do
    python train.py --model_name_or_path google/t5-$size-lm-adapt --output_dir runs_large/$data/42/t5-$size-lm-adapt --dataset_name mainlp/"$data"_donkii --do_train --overwrite_output_dir --bf16  --per_device_eval_batch_size 60 --num_train_epochs 10  --save_strategy epoch --save_total_limit -1 --learning_rate 1e-3 --max_target_length 256 --max_source_length 512 --gradient_accumulation_steps 30 --logging_steps 100 --seed 42
done
    python calc_error_scores.py --models runs_large/p3/$seed/t5-$size-lm-adapt/checkpoint-* --data mainlp/"$data"_donkii
done

python build_results_table.py --data p3

data=sni
for seed in 42 43 44 
do
for size in small base large xl
do
    python train.py --model_name_or_path google/t5-$size-lm-adapt --output_dir runs_large/$data/42/t5-$size-lm-adapt -dataset_name mainlp/"$data"_donkii --do_train --overwrite_output_dir --bf16  --per_device_eval_batch_size 60 --num_train_epochs 10  --save_strategy epoch --save_total_limit -1 --learning_rate 1e-3 --max_target_length 256 --max_source_length 768 --gradient_accumulation_steps 30 --logging_steps 100 --seed 42
done
    python calc_error_scores.py --models runs_large/p3/$seed/t5-$size-lm-adapt/checkpoint-* --data mainlp/"$data"_donkii
done

python build_results_table.py --data sni

data=adc
for seed in 42 43 44
do
for size in small base large xl
do
    python train.py --model_name_or_path google/t5-$size-lm-adapt --output_dir runs_large/$data/42/t5-$size-lm-adapt --dataset_name mainlp/"$data"_donkii --do_train --overwrite_output_dir --bf16  --per_device_eval_batch_size 60 --num_train_epochs 10  --save_strategy epoch --save_total_limit -1 --learning_rate 1e-3 --max_target_length 256 --max_source_length 768 --gradient_accumulation_steps 30 --logging_steps 100 --seed 42
done
    python calc_error_scores.py --models runs_large/p3/$seed/t5-$size-lm-adapt/checkpoint-* --data mainlp/"$data"_donkii
done

python build_results_table.py --data adc
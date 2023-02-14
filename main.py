import torch
import warnings
warnings.filterwarnings("ignore")
from utils import train_tokenizer,save_src
import transformers

import numpy as np
import os
import json
import sacrebleu
# huggingface
from transformers import set_seed
from transformers import EarlyStoppingCallback
# own
import CONSTANT
from model import MarianMTModel
from data import get_datasets,Tokenizer, get_datasets
from _transformers import MySeq2SeqTrainer
from args import DataArgs, TrainingArgs,check_args,MarianConfig
from optim import Adam,get_inverse_sqrt_schedule_with_warmup

def main(data_args,model_args,training_args):
    print(data_args,model_args,training_args,sep='\n')
    set_seed(training_args.seed)

    if data_args.train_tokenizer:
        src_tokenizer = train_tokenizer(
            data_ls = [data_args.train_file,data_args.test_file],
            dict_key = data_args.src,
            vocab_size=model_args.src_vocab_size,
        )

        trg_tokenizer = train_tokenizer(
            data_ls = [data_args.train_file,data_args.test_file],
            dict_key = data_args.trg,
            vocab_size=model_args.trg_vocab_size,
        )
    if model_args.use_joint_bpe:
        trg_tokenizer = Tokenizer(data_args.trg_vocab_file,max_length=model_args.max_trg_len)
        src_tokenizer = Tokenizer(data_args.trg_vocab_file,max_length=model_args.max_src_len) 
        model_args.src_vocab_size = src_tokenizer.vocab_size
        model_args.trg_vocab_size = trg_tokenizer.vocab_size
        model_args.vocab_size = trg_tokenizer.vocab_size # for decoder pre-softmax layer calculating loss
    else:
        src_tokenizer = Tokenizer(data_args.src_vocab_file,max_length=model_args.max_trg_len)
        trg_tokenizer = Tokenizer(data_args.trg_vocab_file,max_length=model_args.max_trg_len)
        model_args.src_vocab_size = src_tokenizer.vocab_size
        model_args.trg_vocab_size = trg_tokenizer.vocab_size
        model_args.vocab_size = trg_tokenizer.vocab_size # for decoder pre-softmax layer calculating loss

    print("Initializing Model...")
    model = MarianMTModel(model_args)
    train_dataset,dev_dataset,test_dataset = get_datasets(data_args,src_tokenizer,trg_tokenizer)
    print("Loading Complete")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = trg_tokenizer.decode_batch(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, trg_tokenizer.pad_token_id)
        decoded_labels = trg_tokenizer.decode_batch(labels, skip_special_tokens=True)

        hyps = [pred.strip() for pred in decoded_preds]
        refs = [label.strip() for label in decoded_labels]
        
        bleu = sacrebleu.corpus_bleu(hyps,[refs],
                                    force=True,lowercase=False,tokenize='none').score
        result = {"bleu":bleu}
        return result


    optimizer = Adam([{'params':model.parameters(), 'lr': model_args.d_model**-0.5}], betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_inverse_sqrt_schedule_with_warmup(optimizer, training_args.warmup_steps, training_args.max_steps)

    

    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        optimizers = (optimizer,lr_schedule),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Training
    if training_args.do_train:
            
            train_result = trainer.train()
            trainer.save_model()  

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()


    if training_args.do_predict:
        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=model_args.max_length,
            num_beams=model_args.num_beams,
        )
        metrics = predict_results.metrics
        max_test_samples = (
            data_args.max_predict_samples if data_args.max_test_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = trg_tokenizer.decode_batch(predict_results.predictions, skip_special_tokens=True)
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    all_results_dir = os.path.join(training_args.output_dir,"all_results.json")
    best_bleu = json.load(open(all_results_dir))["predict_bleu"]
    best_bleu = str(model_args.contrastive_temperature) + '_' + str(best_bleu)
    log_dir = "/".join(training_args.output_dir.split('/')[:-1])
    if os.system(f"mv {training_args.output_dir} {os.path.join(log_dir,best_bleu)}") != 0:
        raise RuntimeError("Rename Unsuccessful")

if __name__ == '__main__':
    
    data_args,model_args,training_args = check_args(DataArgs(),MarianConfig(),TrainingArgs())
    main(data_args,model_args,training_args)



import copy
import torch
import argparse

# Get device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Dictionary of all subtasks by task category
categories = {'QG': ['subtask001_quoref_question_generation', 
                     'subtask003_mctaco_question_generation_event_duration', 
                     'subtask006_mctaco_question_generation_transient_stationary', 
                     'subtask009_mctaco_question_generation_event_ordering',
                     'subtask012_mctaco_question_generation_absolute_timepoint',
                     'subtask015_mctaco_question_generation_frequency',
                     'subtask023_cosmosqa_question_generation',
                     'subtask026_drop_question_generation',
                     'subtask031_winogrande_question_generation_object',
                     'subtask032_winogrande_question_generation_person',
                     'subtask040_qasc_question_generation',
                     'subtask048_multirc_question_generation',
                     'subtask060_ropes_question_generation4'],
              'AG': ['subtask002_quoref_answer_generation', 
                     'subtask004_mctaco_answer_generation_event_duration', 
                     'subtask007_mctaco_answer_generation_transient_stationary',
                     'subtask010_mctaco_answer_generation_event_ordering',
                     'subtask013_mctaco_answer_generation_absolute_timepoint',
                     'subtask016_mctaco_answer_generation_frequency',
                     'subtask024_cosmosqa_answer_generation',
                     'subtask028_drop_answer_generation',
                     'subtask033_winogrande_answer_generation',
                     'subtask041_qasc_answer_generation',
                     'subtask043_essential_terms_answering_incomplete_questions',
                     'subtask047_misc_answering_science_questions',
                     'subtask051_multirc_correct_answer_single_sentence',
                     'subtask054_multirc_write_correct_answer',
                     'subtask058_multirc_question_answering',
                     'subtask061_ropes_answer_generation4'],
              'IAG': ['subtask005_mctaco_wrong_answer_generation_event_duration', 
                      'subtask008_mctaco_wrong_answer_generation_transient_stationary',
                      'subtask011_mctaco_wrong_answer_generation_event_ordering',
                      'subtask014_mctaco_wrong_answer_generation_absolute_timepoint',
                      'subtask017_mctaco_wrong_answer_generation_frequency',
                      'subtask025_cosmosqa_incorrect_answer_generation',
                      'subtask042_qasc_incorrect_option_generation',
                      'subtask055_multirc_write_incorrect_answer'],
              'CF': ['subtask018_mctaco_temporal_reasoning_presence',
                     'subtask019_mctaco_temporal_reasoning_category',
                     'subtask020_mctaco_span_based_question',
                     'subtask021_mctaco_grammatical_logical',
                     'subtask022_cosmosqa_passage_inappropriate_binary',
                     'subtask027_drop_answer_type_generation',
                     'subtask046_miscellaenous_question_typing',
                     'subtask049_multirc_questions_needed_to_answer',
                     'subtask050_multirc_answerability',
                     'subtask052_multirc_identify_bad_question',
                     'subtask056_multirc_classify_correct_answer',
                     'subtask057_multirc_classify_incorrect_answer',
                     ],
              'MM': ['subtask029_winogrande_full_object',
                     'subtask030_winogrande_full_person',
                     'subtask034_winogrande_question_modification_object',
                     'subtask035_winogrande_question_modification_person',
                     'subtask036_qasc_topic_word_to_generate_related_fact',
                     'subtask037_qasc_generate_related_fact',
                     'subtask038_qasc_combined_fact',
                     'subtask045_miscellaneous_sentence_paraphrasing',
                     'subtask053_multirc_correct_bad_question',
                     'subtask059_ropes_story_generation4'],
              'VF': ['subtask039_qasc_find_overlapping_words',
                     'subtask044_essential_terms_identifying_essential_words',
                     ],
              }

# Instructions Encoding
def no_examples_encoding(task, inp):
    return f"""Definition: {task['Definition']}
Prompt: {task['Prompt']}
Things to Avoid: {task['Things to Avoid']}
Emphasis&Caution: {task['Emphasis & Caution']}
Input: {inp}
Output:"""

from transformers import TrainerCallback

# Logging Callback
class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Fine-Tuning a model on the cross-task generalization dataset")
    parser.add_argument("evaluation_tasks", help="File which contains tasks for evaluation - generate this using AssignTasks.py")
    parser.add_argument("output_directory", help="Name of the folder to save the finetuned model in")
    parser.add_argument("model", help="Which model to finetune - please use t5 or bart")

    args = parser.parse_args()
    config = vars(args)

    # Configure training tasks by removing eval tasks, and categories not used in this model
    trainingPrompts = copy.deepcopy(categories)
    evalTasks = open(config['evaluation_tasks'], 'r')
    lines = evalTasks.readlines()
    keysToUse = []
    for line in lines:
        line = line.split(":")
        keysToUse.append(line[0])
        subtasks = line[1].split(",")[:-1]
        for subtask in subtasks:
            subtask = subtask.strip()
            trainingPrompts[line[0]].remove(subtask)

    evalTasks.close()

    from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

    
    # Initialize model and tokenizer
    if (config['model'] == 't5'):
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
    elif (config['model'] == 'bart'):
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    model.to(device)

    # Create pandas dataframe of samples, using no_examples_encoding, and process data into usable form
    import pandas as pd
    from datasets import Dataset, DatasetDict
    import json

    training_dict = {'Instructions': [], 'Outputs': []}

    # For each subtask in the training set, append inputs and outputs to the dataset
    for category in trainingPrompts.keys():
        # Skip subtasks not being fine-tuned with this model
        if (category not in keysToUse): continue
        for task in trainingPrompts[category]:
            with open('./app_static_tasks_sample/' + task + '.json') as json_file:
                subtask = json.load(json_file)
                for instance in subtask['Instances']:
                    string_encoding = no_examples_encoding(subtask, instance['input'])
                    training_dict['Instructions'].append(string_encoding)
                    training_dict['Outputs'].append(instance['output'][0])
                
    df_training = pd.DataFrame(training_dict)

    # This is only for testing with small datasets, comment out for proper fine-tuning
    #df_training = df_training.sample(5000)

    training_dataset = Dataset.from_pandas(df_training)

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['Instructions'], padding='max_length', max_length=128, truncation=True)
        target_encodings = tokenizer.batch_encode_plus(example_batch['Outputs'], padding='max_length', max_length=128, truncation=True)
    
        labels = target_encodings['input_ids']
    
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
        }

        return encodings
    
    trainingDelimiter = int(len(df_training) * (3/4))

    # Create Dataset in huggingface acceptable format, for now using all tasks(will probably take way too long)
    for_finetuning = DatasetDict(
        train=training_dataset.shuffle(seed=1111).select(range(trainingDelimiter)),
        val=training_dataset.shuffle(seed=1111).select(range(trainingDelimiter, len(df_training)))
    )

    tokenized_data = for_finetuning.map(
        convert_to_features,
        batched=True,
        batch_size=16
    )

    tokenized_data = tokenized_data.remove_columns(["Instructions"])
    tokenized_data.set_format("torch")

    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    arguments = Seq2SeqTrainingArguments(
        output_dir=config['output_directory'],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['val'],
        tokenizer=tokenizer,
    )

    trainer.add_callback(LoggingCallback(f"{config['output_directory']}/log.jsonl"))

    train_result = trainer.train()

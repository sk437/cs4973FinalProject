import copy
import random
import argparse


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

parser = argparse.ArgumentParser(description="Generate a txt file containing a set of random tasks to use for evaluation - specify each type of task which should be included.")
parser.add_argument("output_file", help="The name of the file to store this evaluation configuration")
parser.add_argument("--QG", action="store_true", help="Include Question Generation")
parser.add_argument("--AG", action="store_true", help="Include Answer Generation")
parser.add_argument("--IAG", action="store_true", help="Include Incorrect Answer Generation")
parser.add_argument("--CF", action="store_true", help="Include Classification")
parser.add_argument("--MM", action="store_true", help="Include Minimal Text Modification")
parser.add_argument("--VF", action="store_true", help="Include Verification")


args = parser.parse_args()
config = vars(args)

# Move two of each specified task into the evaluation category

out = open(config['output_file'], 'w')

#out.write(f'QG:{config["QG"]}, AG:{config["AG"]}, IAG:{config["IAG"]}, CF:{config["CF"]}, MM:{config["MM"]}, VF:{config["VF"]} \n')

# Get a random set of tasks for evaluation
trainingPrompts = copy.deepcopy(categories)
evaluationPrompts = {'QG': [], 'AG': [], 'IAG': [], 'CF': [], 'MM': [], 'VF': []}
for key in trainingPrompts.keys():
    if (not config[key]): continue;
    subtask = random.choice(trainingPrompts[key])
    trainingPrompts[key].remove(subtask)
    evaluationPrompts[key].append(subtask)
    subtask = random.choice(trainingPrompts[key])
    trainingPrompts[key].remove(subtask)
    evaluationPrompts[key].append(subtask)

# Write these tasks to the file
for key in evaluationPrompts.keys():
    if (not config[key]): continue;
    out.write(key + ":")
    for task in evaluationPrompts[key]:
        out.write(task + ",")
    out.write("\n")

out.close()
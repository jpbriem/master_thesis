################## General Task Explanation ##################
general_explanation = '''You are confronted with a task in which a 1-dimensional input sequence of pixels should be transformed into a corresponding output sequence. The input and output sequences have values from 'a' to 'i' representing different colors, and '.' representing the background color. Adjacent pixels of the same color are designated as objects. For example ['.','b','b','.','c'] represents a pixel sequence with the following objects: Object_1: {{color: 'b', position: [1,2], size: 2}}, Object_2: {{color: 'c', position: [4], size: 1}}, with zero-indexing for the position.\n'''
human_priors = '''\nThe transformation from input to output follows a certain pattern with logical rules that might refer to concepts as follows:
- Geometry: Symmetries, mirroring, connecting points.
- Objects: 
	- transformations, such as move, hollow, scale, remove, copy, recolor.
	- relations between objects, such as distance, alignment, overlap, containment.
- Noise pixels.
- Arithmetics based on objects: Counting, sorting.
- Conditions: rules might be conditional.
This list is not exhaustive.'''

################## Prompt Templates ##########################

standard_prompt = {
	"user": '''{context}{test_input}'''
}

cot_prompt = {
    "system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }


vote_prompt = {
    "system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }

value_prompt = {
    "system": general_explanation + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }

failure_analysis_prompt = {
    "system": general_explanation + '''\nYou are given an input sequence and 2 output sequences, one is wrong and the other is the ground truth. Your task is to compare the output sequences.\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''Input: {test_input}Output ground truth: {output_gt}Output wrong: {output_wrong}'''
 }

revision_prompt = {
	"system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{hypotheses}'''
}

compare_prompt = '''

'''

score_prompt = '''

'''

################## Prompt modules per step ##################
# new try - revised prompts!
prompt_modules = {
 	"0": { # description 
		'spread': True,
		'phase': 'abstraction',
  		'generation': {
			"instruct_task": f'\n\nYour task is to describe the objects in the given input and output sequences.',
			"output_format": {
				'objects': {
					'Example_1': {
						'input': 'regarding the first example, identify all objects in the input sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
						'output': 'regarding the first example, identify all objects in the output sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
						},
					'Example_2': {...},
    				}
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nYou are given example input-output pairs with descriptions about identified objects.',
			"instruct_task": f'\n\nEvaluate the given object descriptions and analyze if they correctly cover all objects. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'input_analysis': 'Regarding the first example, analyze if the given object descriptions correctly cover all objects in the input sequence.',
                    'output_analysis': 'Regarding the first example, analyze if the given object descriptions correctly cover all objects in the output sequence',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given object descriptions as integer.'
                    },
                'Example_2': {...},
                }
   		 	}
     	},
 	"1": { # pattern 
		'spread': True,
		'phase': 'abstraction',
  		'generation': {
        	"instruct_task": f'\n\nImagine you want to explain how to transform a new input without knowing its output yet. Your task is to infer the overall pattern that describes the relation between all input-output pairs.',
			"output_format": {
				'Example_1': {
					'object_number': 'analyze if and how the number of objects changed from input to output',
					'object_analysis': 'make an in-depth analysis and compare the input and output objects, focus on color, position, size',
					'object_relations': 'can you identify relationships between objects from the input and objects from the output?',
					'object_transformation': 'based on the input, how can we determine the output object\'s color, position, and size? Focus on conditions explaining the transformation',
					},
				'Example_2': {...},
				'overall_pattern': { 
					'conditions': 'why do objects change? Search for conditions in the input based on object colors, positions, and sizes!',
					'overall_pattern': 'combine your findings and describe general rules to transform inputs into outputs valid for all examples, focusing on WHAT type of object changed WHY and HOW. Be specific!', 
					},
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that describes the relation between the input and output sequences of all examples.',
			"instruct_task": f'\n\nEvaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'conditions_analysis': 'Regarding the first example, analyze if the given conditions refer only to the input and are relevant to determine the object changes.',
                    'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern describes the transformation from input to output.',
                    'precision_analysis': 'Regarding the first example, analyze if the given overall pattern is precise enough to describe the transformation from input to output.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given hints and pattern as integer.'
                    },
                'Example_2': {...},
                }
   		 	}
     	}, 
	"2": { # instructions/algorithm
		'spread': True,
		'phase': 'abstraction',
  		'generation': {
			"instruct_task": f'\n\nYour task is to give a textual step-by-step transformation algorithm that is generally applicable to all examples to transform the input sequence into its output sequence.',
			"output_format": {
				'conditions': 'list all relevant conditions regarding the input that determine the transformation',
				'transformation_algorithm': 'create a textual transformation algorithm that is generally applicable to transform a given input sequence into its output sequence, focus on conditions. Be specific!',
   		 		},
			},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a step-by-step transformation algorithm that transforms an input sequence into its output sequence.',
			"instruct_task": f'\n\nEvaluate the given algorithm and analyze if it correctly describes the transformation for all examples. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'algorithm_analysis': 'Regarding the first example, analyze if the algorithm correctly transforms the input sequence into its output sequence.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the algorithm as integer.'
                    },
                'Example_2': {...},
                },
   		 	},
     	},
	"3": { # test case
		'spread': True,
		'phase': 'application',
  		'generation': {
        	"instruct_task": f'\n\nMoreover, you are given a new test case with a new input sequence. Your task is to transform the test input sequence into its test output sequence.',
			"output_format": {
				'input_description': 'identify all objects in the input sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
                'algorithm_execution': 'apply the algorithm step-by-step to the test input sequence; focus on potential transformation conditions and respond to every algorithm step in detail.',
                'output': 'return only the resulting test output sequence as numpy array' 
                }
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a test input sequence and a potential test output sequence.',
			"instruct_task": f'\n\nEvaluate the given test output sequence and analyze if the transformation algorithm was applied correctly to the test input sequence. Be as critical as possible with all details!',
			"output_format": {
                'test_output_analysis': 'consider each step of the transformation algorithm and analyze if the test input sequence was correctly transformed into its test output sequence.',
                'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
                }
   		 	},
     	'revision': {
			'analysis': {
				'output_format': {
					'input_objects': 'identify all objects in the input sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
					'output_wrong_objects': 'identify all objects in the wrong output sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
					'output_gt_objects': 'identify all objects in the ground truth output sequence by following the format: "Object_ID: {color: \'object color\', position: [start index, end index], size: number of pixels}".',
					'comparison': 'compare the wrong output to the ground truth and identify all differences, focusing on sequence length and objects.',
					'potential_mistakes': 'analyse the identified differences and make 3 hypotheses about potential mistakes in the transformation process from input to output. Be specific!'
					}
       			},
   			'revision':  {
				'instruct_task': f'Moreover, you are given potential causes of mistakes in the pattern and instructions.\n\nHowever, the given overall pattern is wrong and your task is to correct and revise the overall pattern.',
				'output_format': {
					'pattern_analysis': 'analyse the given wrong overall pattern with respect to the potential mistakes',
					'potential_modification': 'brainstorm about opportunities to modify the overall pattern to correct the mistakes',
					'revision': {
						'overall_pattern': 'write down in detail the complete revised overall pattern',
						'transformation_algorithm': 'write down in detail the complete revised algorithm to transform inputs into outputs'
						}
					}
       			}
			}
      	}
   }

# old one
# prompt_modules = {
# 	"0": {
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to give only an abstract description about how an input sequence and how an output sequence typically look like based on the examples.',
# 			"output_format": {
#                 'sequence_view': 'describe the dimensions of all input sequences and of all output sequences one after another.', 
#                 'pixel_view': 'describe the pixels of all input sequences and of all output sequences one after another, focusing on positions, patterns, or counts', 
#                 'object_view': 'describe the objects in all input sequences and in all output sequences one after another, focusing on shape, amount, size, position, values, cell count', 
# 				'description': {	
#      				'input_description': 'Regarding all input sequences, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical input sequence has a dimension of ... and looks like..."',
# 					'output_description': 'Regarding all output sequences, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical output sequence has a dimension of ... and looks like..."',
# 					},
#     			},
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an abstract description about how an input sequence and an output sequence typically look like.',
# 			"instruct_task": f'\n\nEvaluate the given description and analyze if it correctly describes the provided input and output sequences of all examples.',
# 			"output_format": {
#                 'Example_1': {
#                     'description_analysis': 'Regarding the first example, analyze if both the input and output sequence fit to the given description.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the description as integer.'
#                     },
#                 'Example_2': {
#                     'description_analysis': '...',
#                     'value': '...'
#                     },
#                 }
#    		 	}
#      	},
#  	"1": {
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to infer an overall pattern that describes the simplest relation between all input and output pairs.',
# 			"output_format": {
#                 'sequence_changes': 'For each example: describe if and how the dimension of the input sequences is different from its output sequence', 
#                 'pixel_changes': 'For each example: describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'For each example: describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that might describe the relation between the input and output sequences of all examples.',
# 			"instruct_task": f'\n\nEvaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples.',
# 			"output_format": {
#                 'Example_1': {
#                     'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern correctly describes the relation between the input and output.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the overall pattern as integer.'
#                     },
#                 'Example_2': {
#                     'overall_pattern_analysis': '...',
#                     'value': '...'
#                     },
#                 }
#    		 	}
#      	},
# 	"2": {
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to give step-by-step instructions that are general applicable to all examples to get from the input sequence to its output sequence.',
# 			"output_format": {
#                 'part_of_interest': 'regarding the transformation, describe the parts of interest of the input sequence, e.g. the sequence dimension, pixel pattern, or objects',
#                 'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
#                 'instructions': 'describe all transformation steps with potential conditions and provide step-by-step instructions that are general applicable to transform an input sequence into its output sequence', 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given step-by-step instructions that are generally applicable to transform an input sequence into its output sequence.',
# 			"instruct_task": f'\n\nEvaluate the given set of instructions and analyze if it correctly describes the transformation for all examples.',
# 			"output_format": {
#                 'Example_1': {
#                     'instruction_analysis': 'Regarding the first example, analyze if the given instructions correctly transform the input sequence into its output sequence. ',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
#                     },
#                 'Example_2': {
#                     'instruction_analysis': '...',
#                     'value': '...'
#                     },
#                 }
#    		 	}
#      	},
# 	"3": {
# 		'generation': {
# 			"instruct_task": f'\n\nMoreover, you are given a new test case with an input sequence. Your task is to transform the test input sequence into its test output sequence.',
# 			"output_format": {
#                 'description': 'describe the test input and check if it fits to the given abstract description',
#                 'intermediate_results': 'apply the instructions step-by-step to the test input sequence; focus on potential transformation conditions and provide all intermediate sequences',
#                 'output': 'return only the resulting test output sequence as numpy array' 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a test input sequence and a potential test output sequence.',
# 			"instruct_task": f'\n\nEvaluate the given test output sequence and analyze if it fits to the given description, overall pattern, and instructions.',
# 			"output_format": {
#                 'test_output_analysis': 'analyze if the given test output fits to the given description, overall pattern, and instructions.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
#                 }
#    		 	}
#      	},
#    }

################## Prompt modules Single CoT ################

prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the simplest possible relation beetween input and output. Then, your task is to transform the test input sequence into its test output sequence.',
			"output_format": {
          		'sequence_changes': 'describe if the dimension of the input sequence is different to its output sequence', 
				'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
				'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
				'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
				'instructions': 'describe the transformation actions in detail step by step', 
				'test_output': 'Use the instructions to transform the test input sequence and return only the resulting output sequence in numpy array format.'
            	},
   		 	},
		},
	}

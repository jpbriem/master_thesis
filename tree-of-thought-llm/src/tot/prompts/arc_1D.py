################## General Task Explanation ##################
general_explanation = '''You are confronted with a task in which a 1-dimensional input sequence of pixels should be transformed into a corresponding output sequence. The input and output sequences have values from 1 to 9 representing different pixel colors, and 0 representing the background color. Adjacent pixels of the same color are designated as objects. For example [0, 2, 2, 0, 3] represents a pixel sequence of dimension [1, 5] with the following objects: [Object_1: {{'color': 2, 'start_index': 1, 'end_index': 2, 'size': 2}}, Object_2: {{'color': 3, 'start_index': 4, 'end_index': 4, 'size': 1}}], with zero-indexing for the position.\n'''

human_priors = '''\nThe transformation from input to output follows a certain pattern with logical rules that might refer to concepts as follows:
- Objects: 
	- transformations, such as move, hollow, scale, remove, copy, recolor.
	- relations between objects, such as distance, alignment, overlap, containment.
- Noise pixels.
- Arithmetics based on objects: Counting, sorting.
- Conditions: rules might be conditional.
This list is not exhaustive.'''

################## Prompt Templates ##########################

standard_prompt = {
	"user": '''{context}{test_input}''' # \n\nGive no explanation. 
}

cot_prompt = {
    "system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format, fill the values as described: {output}. Do not use quotation marks ' or " within the fields.\n''',
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

# no object detection tool 
prompt_modules = {
 	"0": { # description 
		'spread': True,
		'phase': 'abstraction', 
  		'generation': {
			"instruct_task": f'\n\nYour task is to describe the objects in the given input and output sequences.',
			"output_format": {
				'Example_1': {
					'input': 'regarding the first example, describe all objects in the input sequence.',
					'output': 'regarding the first example, describe all objects in the output sequence.',
					},
				'Example_2': {...},
				'description': {
        			'input': 'summarize your findings to highlight commonalities within input sequences.',
        			'output': 'summarize your findings to highlight commonalities within output sequences.',
					},
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nYou are given example input-output pairs with descriptions, detailing similarities unique to inputs and outputs respectively.',
			"instruct_task": f'\n\nEvaluate the given descriptions and analyze if they fit to the examples and cover all relevant commonalities. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'input_analysis': 'Regarding the first example, analyze if the given input description fits to the example and covers all relevant commonalities with other inputs.',
                    'output_analysis': 'Regarding the first example, analyze if the given output description fits to the example and covers all relevant commonalities with other outputs.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given descriptions as integer.'
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
					'object_relations': 'can you identify relationships between objects from the input that became objects from the output?',
					'object_transformation': 'based on the input, how can we determine the output object\'s color, position, and size? Focus on conditions explaining the transformation',
					},
				'Example_2': {...},
				'overall_pattern': { 
					'conditions': 'why do objects change? Search for conditions in the input based on object colors, positions, and sizes!',
					'overall_pattern': 'define general rules to transform any input into its output based only on the input. Specify WHAT type of object changes, WHY it changes, and HOW. Be specific!',
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
                    'precision_analysis': 'Regarding the first example, analyze if the given overall pattern is precise enough to transform a new input to its output.',
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
			"instruct_task": f'\n\nYour task is to give detailed transformation steps that are generally applicable to all examples to transform the input sequence into its output sequence.',
			"output_format": {
				'conditions': 'list all relevant conditions regarding the input that determine the transformation',
				'transformation_steps': 'create a list of detailed transformation steps that are generally applicable to transform a given input sequence into its output sequence, focus on conditions. Be specific!',
   		 		},
			},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a list of detailed transformation steps that transform an input sequence into its output sequence.',
			"instruct_task": f'\n\nEvaluate the given transformation steps and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'transformation_analysis': 'Regarding the first example, analyze if the transformation steps correctly transform the input sequence into its output sequence.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the transformation steps as integer.'
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
				'input_description': 'regarding the test input, describe the objects in the input sequence, focusing on size, position, color.',
               	'transformation': 'apply the transformation steps to the test input sequence, detailing how each condition of the transformation pattern applies to the current task and respond to every step in detail.',
				'transformed_objects': 'describe how the objects should look like in the test output sequence, focusing on size, position, color',
                'output': 'return only the resulting test output sequence as numpy array' 
                },   
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a test input sequence and a potential test output sequence.',
			"instruct_task": f'\n\nEvaluate the given test output sequence and analyze if the transformation was applied correctly to the test input sequence. Be as critical as possible with all details!',
			"output_format": {
                'test_output_analysis': 'consider each step of the transformation algorithm and analyze if the test input sequence was correctly transformed into its test output sequence.',
                'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer. Return 0 if no test output sequence was given.'
                }
   		 	},
     	'revision': {
			'analysis': {
				'output_format': {
					'input_objects': 'identify all objects in the input sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
					'output_gt_objects': 'identify all objects in the ground truth output sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
					'output_wrong_objects': 'identify all objects in the wrong output sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
					'comparison': 'compare the wrong output to the ground truth and identify all differences, focusing on sequence length and objects.',
					'potential_mistakes': 'analyse the identified differences and make 3 hypotheses about potential mistakes in the transformation process from input to output. Be specific!'
					}
       			},
   			'revision':  {
				'instruct_task': f'\nMoreover, you are given potential causes of mistakes in the pattern and instructions.\n\nHowever, the given overall pattern is wrong and your task is to correct and revise the overall pattern.',
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


# #  for object representation
# prompt_modules = {
#  	"0": { # pattern 
# 		'spread': True,
# 		'phase': 'abstraction',
#   		'generation': {
#         	"instruct_task": f'\n\nImagine you want to explain how to transform a new input without knowing its output yet. Your task is to infer the overall pattern that describes the relation between all input-output pairs.',
# 			"output_format": {
# 				'Example_1': {
# 					'object_number': 'regarding the first example, analyze if and how the number of objects changed from input to output',
# 					'object_analysis': 'regarding the first example, describe the differences between the input and output objects, be precise and say WHAT changed HOW, focus on color, position, size',
# 					'conditions': 'regarding the first example, why do certain objects change? Search for conditions in the input that determine the changes, focus on object colors, positions, and sizes!',
# 					},
# 				'Example_2': {...},
# 				'overall_pattern': { 
#         			'conditions': 'regarding all examples, why do certain objects change? Search for conditions in the inputs that determine the changes, focus on object colors, positions, and sizes!',
# 					'overall_pattern': 'define general rules to transform any input into its output based only on the input. Specify WHAT type of object changes, WHY it changes, and HOW. Be specific!',
#      				},
#                 },
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that describes the relation between the inputs and outputs of all examples.',
# 			"instruct_task": f'\n\nEvaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples. Be as critical as possible with all details!',
# 			"output_format": {
#                 'Example_1': {
#                     'conditions_analysis': 'Regarding the first example, analyze if the given conditions refer only to the input and are relevant to determine the object changes.',
#                     'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern describes the transformation from input to output.',
#                     'precision_analysis': 'Regarding the first example, analyze if the given overall pattern is precise enough to transform a new input to its output.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given hints and pattern as integer.'
#                     },
#                 'Example_2': {...},
#                 }
#    		 	}
#      	}, 
# 	"1": { # instructions/algorithm
# 		'spread': False,
# 		'phase': 'abstraction',
#   		'generation': {
# 			"instruct_task": f'\n\nYour task is to give detailed transformation steps that are generally applicable to all examples to transform the input into its output.',
# 			"output_format": {
# 				'conditions': 'list all relevant conditions regarding the input that determine the transformation',
# 				'transformation_steps': 'create a list of detailed transformation steps that are generally applicable to transform a given input into its output, focus on conditions. Be specific!',
#    		 		},
# 			},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a list of detailed transformation steps that transform an input into its output.',
# 			"instruct_task": f'\n\nEvaluate the given transformation steps and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all details!',
# 			"output_format": {
#                 'Example_1': {
#                     'transformation_analysis': 'Regarding the first example, analyze if the transformation steps correctly transform the input into its output.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the transformation steps as integer.'
#                     },
#                 'Example_2': {...},
#                 },
#    		 	},
#      	},
# 	"2": { # test case
# 		'spread': True,
# 		'phase': 'application',
#   		'generation': {
#         	"instruct_task": f'\n\nMoreover, you are given a new test case with a new input. Your task is to transform the test input into its test output.',
# 			"output_format": {
# 				'input_description': 'regarding the test input, describe the objects in the input, focusing on size, position, color.',
#                	'transformation': 'apply the transformation steps and describe in natural language how the objects should look like in the test output, focusing on size, position, color',
# 				'output': {
# 					'test_case_output_dimension': 'state the dimension of the test case output [rows, columns] as list of integers',
# 					'transformed_objects': 'Describe the transformed objects for the test output by following the format in the test case input.',
# 					},
#                 },   
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a test input and a potential test output.',
# 			"instruct_task": f'\n\nEvaluate the given test output and analyze if the transformation was applied correctly to the test input. Be as critical as possible with all details!',
# 			"output_format": {
#                 'test_output_analysis': 'consider each step of the transformation instructions and analyze if the test input was correctly transformed into its test output.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer. Return 0 if no test output was given.'
#                 }
#    		 	},
#      	'revision': {
# 			'analysis': {
# 				'output_format': {
# 					'input_objects': 'identify all objects in the input sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
# 					'output_gt_objects': 'identify all objects in the ground truth output sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
# 					'output_wrong_objects': 'identify all objects in the wrong output sequence by following the format: "Object_ID: {color: \'object color\', start_index: i, end index: j, size: number of pixels}".',
# 					'comparison': 'compare the wrong output to the ground truth and identify all differences, focusing on sequence length and objects.',
# 					'potential_mistakes': 'analyse the identified differences and make 3 hypotheses about potential mistakes in the transformation process from input to output. Be specific!'
# 					}
#        			},
#    			'revision':  {
# 				'instruct_task': f'\nMoreover, you are given potential causes of mistakes in the pattern and instructions.\n\nHowever, the given overall pattern is wrong and your task is to correct and revise the overall pattern.',
# 				'output_format': {
# 					'pattern_analysis': 'analyse the given wrong overall pattern with respect to the potential mistakes',
# 					'potential_modification': 'brainstorm about opportunities to modify the overall pattern to correct the mistakes',
# 					'revision': {
# 						'overall_pattern': 'write down in detail the complete revised overall pattern',
# 						'transformation_algorithm': 'write down in detail the complete revised algorithm to transform inputs into outputs'
# 						}
# 					}
#        			}
# 			}
#       	}
#    }




################## Prompt modules Single CoT ################
# no object detection tool 
prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the simplest possible relation between input and output. Then, your task is to transform the test input sequence into its test output sequence.',
			"output_format": {
				'description': {
					'Example_1': 'regarding the first example, describe the difference between the input and output sequence, be precise and say WHAT changed HOW!',
					'Example_2': '...',
    				},
    			'overall_pattern': 'describe the input-output relationship for all input-output pairs', 
				'instructions': 'describe the needed transformation actions to transform a new input into its output, think step by step', 
				'transformation': {
        			'input': 'copy the test case input sequence from the task. Mind the sequence length!',
					'object_description': 'regarding the test input, describe the objects in the input sequences, focusing on size, position, color',
					'transformed_objects': 'Describe how the objects should look like in the test output sequence, focusing on size, position, color',
            		'output': 'create the resulting test output sequence. Mind the sequence length!'
                 	},
            	'test_output': 'Return the created test output sequence in numpy array format. Mind the sequence length!'
             	},
   		 	},
		},
	} 
# for object detection tool
prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the simplest possible relation between input and output. Then, your task is to transform the test input sequence into its test output sequence.',
			"output_format": {
				'description': {
					'Example_1': 'regarding the first example, describe the differences between the input and output objects, be precise and say WHAT changed HOW!',
					'Example_2': '...',
    				},
    			'overall_pattern': 'describe the input-output relationship for all input-output pairs', 
				'instructions': 'describe the needed transformation actions to transform a new input into its output, think step by step', 
				'test_case_input_objects': 'copy the objects of the test case input sequence from the task',
				'transformation': 'Describe in natural language how the objects should look like in the test output sequence, focusing on size, position, color',
				'transformed_objects': 'Describe the transformed objects for the test output sequence by following the format in the test case input.',
				'test_case_output_dimension': 'state the dimension of the test case output sequence [rows, columns] as list of integers',
				'test_case_output': 'Create the test case output pixel sequence with the transformed objects as numpy array, e.g. \"[0, 0, ..., 0]\". Use zero-indexing for the object positions and fill unoccupied cells with the background color!'
             	},
   		 	},
		},
	} 


################## Few-Shot Examples ################

few_shot_ex = None
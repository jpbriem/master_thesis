################## General Task Explanation ##################
general_explanation = '''You are given a series of example input and output pairs that share the same logical pattern of getting the output from its input. Each input and output is a 1-dimensional sequence of pixels. The values from 'a' to 'j' represent different colors, where 'a' represents the background. For example, ['a','b','b','a','a'] represents a sequence of length 5 with color 'b' at index 1 and 2, with zero-indexing.
'''

################## Prompt Templates ##########################

standard_prompt = {
	"user": '''{context}{test_input}'''
}

cot_prompt = {
    "system": general_explanation + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }


vote_prompt = {
    "system": general_explanation + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }

value_prompt = {
    "system": general_explanation + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }

compare_prompt = '''

'''

score_prompt = '''

'''

################## Prompt modules per step ##################
# new try - revised prompts!
prompt_modules = {
 	"0": {
		# pattern 
		'generation': {
			"instruct_task": f'\n\nYour task is to infer an overall pattern that describes the simplest relation between all input and output pairs.',
			"output_format": {
				'Example_1': {
					'pixel_changes': 'describe the changes between the input and output pixels, focusing on color, index, patterns, counts', 
					'object_changes': 'describe the changes between the objects in the input and output sequences, focusing on amount, size, position, counts, symmetry', 
					'parts_of_interest': 'regarding the transformation from input to output, describe and explain the importance of the parts of interest of the input sequence, e.g. pixel pattern, or objects; be specific and describe the parts appropriately (position, color, size, count, symmetry, etc.)',
					},
				'Example_2': {...},
				'overall_pattern': {
					'parts_of_interest': 'Regarding all examples, is there a pattern about how to determine the parts of interest of the input sequence?',
					'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
					},
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given hints to determine parts of interest of the input sequence and an overall pattern that describes the relation between the input and output sequences of all examples.',
			"instruct_task": f'\n\nEvaluate the given hints and pattern and analyze if they correctly describe the relation between the inputs and outputs of all examples. Be as critical as possible with all pattern details!',
			"output_format": {
                'Example_1': {
                    'parts_of_interest_analysis': 'Regarding the first example, analyze if the given hints about parts of interest help to determine the parts of interest with all needed characteristics of the input.',
                    'overall_pattern_analysis': 'Regarding the first example, divide the pattern in distinct parts and analyze each part individually: 1. is it part of the transformation from input to output? 2. is it specific enough?',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given hints and pattern as integer.'
                    },
                'Example_2': {...},
                }
   		 	}
     	},
	"1": {
		'generation': {
			"instruct_task": f'\n\nYour task is to give step-by-step instructions that are generally applicable to all examples to transform the input sequence into its output sequence.',
			"output_format": {
				'Example_1': {
					'part_of_interest': 'describe the parts of interest of the input sequence; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
					'transformation': 'describe the transformation from input to output step-by-step and refer to the parts of interest',
					'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
					},
				'Example_2': {...},
				'instructions': 'summarize the example transformations and provide step-by-step instructions that are generally applicable to transform an input sequence into its output sequence, focus on potential transformation conditions and how to solve them', 
   		 		},
			},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given step-by-step instructions that are generally applicable to transform an input sequence into its output sequence.',
			"instruct_task": f'\n\nEvaluate the given instructions and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all pattern details!',
			"output_format": {
                'Example_1': {
                    'instruction_analysis': 'Regarding the first example, apply the given instructions to the input sequence and analyze if they correctly transform the input sequence into its output sequence.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
                    },
                'Example_2': {
                    'instruction_analysis': '...',
                    'value': '...'
                    },
                },
   		 	},
     	},
	"2": {
		'generation': {
			"instruct_task": f'\n\nMoreover, you are given a new test case with an input sequence. Your task is to transform the test input sequence into its test output sequence.',
			"output_format": {
                'intermediate_results': 'apply the instructions step-by-step to the test input sequence; focus on potential transformation conditions and provide all intermediate sequences',
                'output': 'return only the resulting test output sequence as numpy array' 
                }
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a test input sequence and a potential test output sequence.',
			"instruct_task": f'\n\nEvaluate the given test output sequence and analyze if it fits to the given description, overall pattern, and instructions. Be as critical as possible with all pattern details!',
			"output_format": {
                'test_output_analysis': 'analyze if the given test output fits to the given description, overall pattern, and instructions.',
                'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
                }
   		 	}
     	},
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

# prompt_modules = {
# 	"0": {
# 		'generation': {
# 			"instruct_task": f'\n\nYou are to infer the simplest possible relation beetween input and output. Then, your task is to transform the test input sequence into its test output sequence.',
# 			"output_format": {
#        			'reflection': 'reflect on the answer', 
#           		'sequence_changes': 'describe if the dimension of the input sequence is different to its output sequence', 
# 				'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
# 				'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
# 				'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
# 				'instructions': 'describe the transformation actions in detail step by step', 
# 				'test_output': 'Use the instructions to transform the test input sequence and return only the resulting output sequence in numpy array format.'
#             	},
#    		 	},
# 		},
# 	}

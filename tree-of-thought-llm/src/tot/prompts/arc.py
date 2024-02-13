################## General Task Explanation ##################
general_explanation = '''You are confronted with a task in which a 2-dimensional input grid of pixels should be transformed into a corresponding output grid. The input and output grids have values from 1 to 9 representing different pixel colors, and 0 representing the background color. The transformation can relate to the entire grid or individual objects in the grid. Objects are usually adjacent pixels of a single color. 
Example: [[0, 2, 2, 0, 3], [0, 2, 0, 0, 0]] represents a pixel grid of dimension (2,5) with the following objects: [Object_1: {{color: '2', coordinates: [(0,1), (0,2), (1,1)], size: 3}}, Object_2: {{color: '3', coordinates: [(0,4)], size: 1}}], with zero-indexing for the coordinates.\n'''

human_priors = '''\nThe logical pattern might refer to concepts as follows:
- Geometry and topology:
	- Lines, rectangular shapes.
	- Connecting points, orthogonal projections.
 	- Symmetries, mirroring, rotations, translations.
	- Shape upscaling or downscaling, elastic distortions.
	- Containing / being contained / being inside or outside of a perimeter.
	- Copying, repeating.
	- Patterns or mosaic based on sections.
- Objects:
	- Objects are shapes based on similar colors or based on surroundings.
	- Object transformations based on geometry and topology.
	- Touching objects have at least one adjacent pixel.
	- Noise pixels.
-  Arithmetics based on objects or shapes:
	- Counting.
	- Sorting.

The list is not exhaustive. Transformations can be conditional.'''

# general_explanation_old = '''You are given a series of example input and output pairs that share the same logical pattern of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'i' represent different colors and '.' represents the background. For example, [['.','b','.'],['.','.','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.

# The logical pattern might refer to concepts as follows:
# - Goal-directedness: input is start and output is end state of process 
# - Geometry & topology:
# 	- Lines, rectangular shapes.
# 	- Symmetries, mirroring, rotations, translations.
# 	- Shape upscaling or downscaling, elastic distortions.
# 	- Containing / being contained / being inside or outside of a perimeter.
# 	- Drawing lines, connecting points, orthogonal projections.
# 	- Copying, repeating.
# 	- Patterns or mosaic based on sections.
# - Objects:
# 	- Objects are shapes based on similar colors or based on surroundings.
# 	- Object transformations based on geometry and topology.
# 	- Touching objects have contact with each other.
# 	- Noise pixels.
# -  Arithmetics based on objects or shapes pixels:
# 	- Counting.
# 	- Sorting.

# The list is not exhaustive. Transformations can be conditional.'''

################## Prompt Templates ##########################

standard_prompt = {
	"user": '''{context}{test_input}\n\nGive no explanation. '''
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
    "system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
 }

failure_analysis_prompt = {
    "system": '''You are confronted with a task in which a 2-dimensional grid of pixels should be transformed. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'i' represent different colors, and '.' represents the background. For example, [['.','b','.'],['.','.','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.\n
You are given an input grid and 2 output grids, one is wrong and the other is the ground truth. Your task is to compare the output sequences.\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''Input: {test_input}\nOutput ground truth: {output_gt}\nOutput wrong: {output_wrong}'''
 }

revision_prompt = {
	"system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''{context}{previous_thoughts}{test_input}'''
}


compare_prompt = '''

'''

score_prompt = '''

'''

################## Prompt modules per step ##################
# TODO: Test
prompt_modules = {
	"0": {
		# grid dimension
		'generation': {
			"instruct_task": f'\n\nYour task is to describe the objects in the given input and output grids.',
			"output_format": {
				'objects': {
					'Example_1': { 
						'input': 'regarding the first example, identify all objects in the input grid by following the format: [Object_ID: {color: \'object color\', coordinates: [(x_1,y_1), (x_2,y_2), ..], size: number of pixels}, ...]',
						'output': 'regarding the first example, identify all objects in the output grid by following the format: [Object_ID: {color: \'object color\', coordinates: [(x_1,y_1), (x_2,y_2), ..], size: number of pixels}, ...]',
					},
					'Example_2': {...},
				},
   		 	},
		},
		'evaluation': {
			"instruct_previous_thoughts": f'\nYou are given example input-output pairs with descriptions about identified objects.',
			"instruct_task": f'\n\nEvaluate the given object descriptions and analyze if they correctly describe all objects. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'input_analysis': 'Regarding the first example, analyze if the given object descriptions correctly cover all objects in the input grid.',
                    'output_analysis': 'Regarding the first example, analyze if the given object descriptions correctly cover all objects in the output grid',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given object descriptions as integer.'
                    },
                'Example_2': {...},
                },
     		},
		},
 	"1": {
		# pattern 
		'generation': {
			"instruct_task": f'\n\nImagine you want to explain how to transform a new input without knowing its output yet. Your task is to infer the overall pattern that describes the relation between all input-output pairs.',
			"output_format": {
				'Example_1': {
					'pixel_changes': 'regarding the first example, describe the changes between the input and output pixels, focusing on color, coordinates, patterns, counts', 
					'object_changes': 'regarding the first example, describe the changes between the objects in the input and output grids, focusing on shape, size, position, values, counts, symmetry', 
					'parts_of_interest': 'regarding the transformation from input to output, describe the parts of interest of the input grid (e.g. a pixel pattern or objects) and explain their importance; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
					},
				'Example_2': {...},
				'overall_pattern': {
					'conditions': 'why do pixels or objects change? Search for conditions in the input based on colors, positions, and sizes!',
					'overall_pattern': 'define general rules to transform any input into its output based only on the input. Specify WHAT changes, WHY it changes, and HOW. Be specific!',
					},
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that describes the relation between the input and output grids of all examples.',
			"instruct_task": f'\n\nEvaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'conditions_analysis': 'Regarding the first example, analyze if the given conditions refer only to the input and are relevant to determine the changes.',
                    'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern describes the transformation from input to output.',
                    'precision_analysis': 'Regarding the first example, analyze if the given overall pattern is precise enough to transform a new input to its output.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given pattern as integer.'
                    },
                'Example_2': {...},
                }
   		 	}
     	},
	"2": {
		'generation': {
			"instruct_task": f'\n\nYour task is to give detailed transformation steps that are generally applicable to all examples to transform the input grid into its output grid.',
			"output_format": {
				'Example_1': {
					'conditions': 'regarding the first example, list all relevant conditions regarding the input that determine the transformation, focusing on shape, size, position, values, counts, symmetry',
					'transformation': 'regarding the first example, describe the transformation steps needed to transform the input grid into its output grid, focus on conditions. Be specific!',
					},
				'Example_2': {...},
				'transformation_steps': 'create a list of detailed transformation steps that are generally applicable to transform a given input grid into its output grid, focus on conditions. Be specific!',
   		 		},
			},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a list of detailed transformation steps that transform an input grid into its output grid.',
			"instruct_task": f'\n\nEvaluate the given transformation steps and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'transformation_analysis': 'Regarding the first example, analyze if the transformation steps correctly transform the input grid into its output grid.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the transformation steps as integer.'
                    },
                'Example_2': {...},
                },
   		 	},
     	},
	"3": {
		'generation': {
        	"instruct_task": f'\n\nMoreover, you are given a new test case with a new input grid. Your task is to transform the test input grid into its test output grid.',
			"output_format": {
                'input_description': 'describe the test input and identify all objects in the input grid by following the format: [Object_ID: {color: \'object color\', coordinates: [(x_1,y_1), (x_2,y_2), ..], size: number of pixels}, ...]',
                'transformation': 'apply the transformation steps to the test input grid, detailing how each condition of the transformation pattern applies to the current task and respond to every step in detail.',
                'output': 'return only the resulting test output grid as numpy array' 
                }
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a test input grid and a potential test output grid.',
			"instruct_task": f'\n\nEvaluate the given test output grid and analyze if the transformation steps were applied correctly to the test input grid. Be as critical as possible with all details!',
			"output_format": {
                'test_output_analysis': 'consider each transformation step and analyze if the test input grid was correctly transformed into its test output grid.',
                'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
                }
   		 	}
     	},
   }




# old with dynamic dimensions, not need here! 
# prompt_modules = {
# 	"0": {
# 		# grid dimension
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to give a description about the dimensions of an input and output grid.',
# 			"output_format": {
# 				'Example_1': {
# 					'grid_view': 'describe the dimension of the input and output grid.', 
# 					'pixel_view': 'describe the pixels of the input and output grid, focusing on shape, amount, size, position, values, counts, symmetry', 
# 					'object_view': 'describe the objects in the input grid and output grid, focusing on shape, amount, size, position, values, cell count', 
# 				},
# 				'Example_2': {...},
#     			'description': {	
#      				'input_dimension': 'Regarding all input grids, is there a pattern describing the typical dimension of an input grid? e.g. all have the same fixed dimension or random or varying dimensions with a pattern..',
# 					'output_dimension': 'Regarding all output grids, is there a pattern describing the typical dimension of an output grid? e.g. all have the same fixed dimension or varying dimensions with a pattern depended on certain characteristics of the input grid; be specific and explain how to determine the output grid dimension!',
# 					},
#     			},
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a description about the dimensions of an input and output grid.',
# 			"instruct_task": f'\n\nEvaluate the given description and analyze if it correctly describes the provided input and output grids of all examples. Be as critical as possible with all pattern details!',
# 			"output_format": {
#                 'Example_1': {
#                     'input_dimension_analysis': 'Regarding the first example, analyze if the input grid fits to the given description.',
#                     'output_dimension_analysis': 'Regarding the first example analyze both: 1. does the output grid fit to the given description? 2. is the description helpful to determine the output grid dimension solely based on the given description and the input grid?',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the description as integer.'
#                     },
#                 'Example_2': {...},
#                 },
#      		},
# 		},
#  	"1": {
# 		# pattern 
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to infer an overall pattern that describes the simplest relation between all input and output pairs.',
# 			"output_format": {
# 				'Example_1': {
# 					'pixel_changes': 'describe the changes between the input and output pixels, focusing on color, coordinates, patterns, counts', 
# 					'object_changes': 'describe the changes between the objects in the input and output grids, focusing on shape, amount, size, position, values, counts, symmetry', 
# 					'parts_of_interest': 'regarding the transformation from input to output, describe and explain the importance of the parts of interest of the input grid, e.g. the grid dimension, pixel pattern, or objects; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
# 					},
# 				'Example_2': {...},
# 				'overall_pattern': {
# 					'parts_of_interest': 'Regarding all examples, is there a pattern about how to determine the parts of interest of the input grid?',
# 					'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
# 					},
#                 },
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given hints to determine parts of interest of the input grid and an overall pattern that describes the relation between the input and output grids of all examples.',
# 			"instruct_task": f'\n\nEvaluate the given hints and pattern and analyze if they correctly describe the relation between the inputs and outputs of all examples. Be as critical as possible with all pattern details!',
# 			"output_format": {
#                 'Example_1': {
#                     'parts_of_interest_analysis': 'Regarding the first example, analyze if the given hints about parts of interest help to determine the parts of interest with all needed characteristics of the input.',
#                     'overall_pattern_analysis': 'Regarding the first example, divide the pattern in distinct parts and analyze each part individually: 1. is it part of the transformation from input to output? 2. is it specific enough?',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given hints and pattern as integer.'
#                     },
#                 'Example_2': {...},
#                 }
#    		 	}
#      	},
# 	"2": {
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to give step-by-step instructions that are general applicable to all examples to transform the input grid into its output grid.',
# 			"output_format": {
# 				'Example_1': {
# 					'part_of_interest': 'describe the parts of interest of the input grid; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
# 					'transformation': 'describe the transformation from input to output step-by-step and refer to the parts of interest',
# 					'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
# 					},
# 				'Example_2': {...},
# 				'instructions': 'summarize the example transformations and provide step-by-step instructions that are generally applicable to transform an input grid into its output grid, focus on potential transformation conditions and how to solve them', 
#    		 		},
# 			},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given step-by-step instructions that are generally applicable to transform an input grid into its output grid.',
# 			"instruct_task": f'\n\nEvaluate the given instructions and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all pattern details!',
# 			"output_format": {
#                 'Example_1': {
#                     'instruction_analysis': 'Regarding the first example, apply the given instructions to the input grid and analyze if they correctly transform the input grid into its output grid.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
#                     },
#                 'Example_2': {
#                     'instruction_analysis': '...',
#                     'value': '...'
#                     },
#                 },
#    		 	},
#      	},
# 	"3": {
# 		'generation': {
# 			"instruct_task": f'\n\nMoreover, you are given a new test case with an input grid. Your task is to transform the test input grid into its test output grid.',
# 			"output_format": {
#                 'description': 'describe the test input with all its parts of interest and try to determine the dimension of the output grid',
#                 'intermediate_results': 'apply the instructions step-by-step to the test input grid; focus on potential transformation conditions and provide all intermediate grids',
#                 'output': 'return only the resulting test output grid as numpy array' 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a test input grid and a potential test output grid.',
# 			"instruct_task": f'\n\nEvaluate the given test output grid and analyze if it fits to the given description, overall pattern, and instructions. Be as critical as possible with all pattern details!',
# 			"output_format": {
#                 'test_output_analysis': 'analyze if the given test output fits to the given description, overall pattern, and instructions.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
#                 }
#    		 	}
#      	},
#    }



################## Prompt modules Single CoT ################

# 1st Try
prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the simplest possible relation between input and output. Then, your task is to transform the test input grid into its test output grid.',
			"output_format": {
          		'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
				'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
				'object_changes': 'describe the changes between the input and output objects, focusing on color, size, coordinates and movement, shape, and object number', 
				'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
				'instructions': 'describe the transformation actions in detail step by step', 
				'test_output': 'Use the instructions to transform the test input grid and return only the resulting output grid in numpy array format.'
            	},
   		 	},
		},
	}
# 2nd Try 
prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the relation between input and output. Then, your task is to transform the test input grid into its test output grid.',
			"output_format": {
				'description': {
					'Example_1': {
						'grid_changes': 'regarding the first example, analyze if the entire grid has changed and describe how',
						#'pixel_changes': 'regarding the first example, describe the changes between the input and output pixels, focusing on movement or pattern changes', 
						'object_changes': 'regarding the first example, describe the changes between the input and output objects, focusing on color, size, coordinates, shape, and object number', 
						},
					'Example_2': {...},
     				},
    			'overall_pattern': 'describe the input-output relationship valid for all input-output pairs', 
				'instructions': 'describe the required transformation actions in detail step by step', 
				'test_case_grid_view': 'regarding the test input, describe the pixels of the entire grid, focusing on patterns', 
#				'pixel_view': 'regarding the test input, describe the pixels, focusing on patterns', 
				'test_case_object_view': 'regarding the test input, describe the objects, focusing on color, size, coordinates and movement, shape, and object number', 
				'test_case_transformation': 'describe how the grid or objects should be transformed',
				'test_case_output': 'create the resulting output grid as numpy array.'	
            	},
   		 	},
		},
	}
# 3rd Try - no nested -> Used for experiments 
prompt_modules_naive = {
	"0": {
		'generation': {
			"instruct_task": f'\n\nYou are to infer the relation between input and output. Then, your task is to transform the test input grid into its test output grid.',
			"output_format": {
				'example_1_description': {
					#'grid_changes': 'regarding the first example, analyze if the entire grid has changed and describe how',
					'pixel_changes': 'regarding the first example, describe the changes between the input and output pixels, focusing on pattern changes', 
					'object_changes': 'regarding the first example, describe the changes between the input and output objects, focusing on color, size, coordinates, shape, and object number', 
					},
				'example_2_description': {...},
    			'overall_pattern': 'describe the input-output relationship valid for all input-output pairs', 
				'instructions': 'describe the required transformation actions in detail step by step', 
				'test_case_input_copy': 'copy the test case input grid from the task',
    			'test_case_grid_view': 'regarding the test input, describe the pixels of the entire grid, focusing on patterns', 
#				'pixel_view': 'regarding the test input, describe the pixels, focusing on patterns', 
				'test_case_object_view': 'regarding the test input, describe the objects, focusing on color, size, coordinates and movement, shape, and object number', 
				'test_case_transformation': 'describe how the grid or objects should be transformed',
				'test_case_output': 'create the resulting output grid as numpy array.'	
            	},
   		 	},
		},
	}



few_shot_ex = None
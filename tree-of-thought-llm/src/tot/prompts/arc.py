################## General Task Explanation ##################
general_explanation = '''You are confronted with a task in which a 2-dimensional input grid of pixels should be transformed into a corresponding output grid. The input and output grids have values from 1 to 9 representing different pixel colors, and 0 representing the background color. The transformation can relate to the entire grid or individual objects in the grid. Objects are usually adjacent pixels of a single color. 
Example: [[0, 2, 2, 0, 3], [0, 2, 0, 0, 2]] represents a pixel grid of dimension [2,5] with the following objects: [Object_1: {{color: 2, coordinates: [[0,1], [0,2], [1,1]], size: 3}}, Object_2: {{color: 3, coordinates: [[0,4]], size: 1}}, Object_3: {{color: '2', coordinates: [[1,4]], size: 1}}], with zero-indexing for the coordinates: [row_index, column_index].\n'''

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

################## Prompt Templates ##########################

standard_prompt = {
	"user": '''{context}{test_input}''' # \n\nGive no explanation. 
}

cot_prompt = {
    "system": general_explanation + human_priors + '''\n{special_instructions}\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the json fields.\n''',
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
# normal
prompt_modules = {
	"0": {
		# grid description
		'spread': True,
		'phase': 'abstraction',
		'generation': {
			"instruct_task": f'\n\nYour task is to describe the given input and output grids.',
			"output_format": {
				'objects': {
					'Example_1': { 
						'input': 'regarding the first example, describe all pixels in the input grid, focusing on pixel coordinates and patterns',
						'output': 'regarding the first example, describe all pixels in the output grid, focusing on pixel coordinates and patterns',
					},
					'Example_2': {...},
				},
				'description': {
					'input': 'summarize your findings to highlight commonalities within input grids by completing the following sentence: "A typical input grid shows pixels that..."',
					'output': 'summarize your findings to highlight commonalities within output grids by completing the following sentence: "A typical output grid shows pixels that..."',
				},
   		 	},
		},
		'evaluation': {
			"instruct_previous_thoughts": f'\nYou are given example input-output pairs with respective descriptions.',
			"instruct_task": f'\n\nEvaluate the given descriptions and analyze if they correctly describe all objects. Be as critical as possible with all details!',
			"output_format": {
                'Example_1': {
                    'input_analysis': 'Regarding the first example, analyze if the given description correctly cover all objects and pixel pattern in the input grid.',
                    'output_analysis': 'Regarding the first example, analyze if the given description correctly cover all objects and pixel pattern in the output grid',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given descriptions as integer.'
                    },
                'Example_2': {...},
                },
     		},
		},
 	"1": {
		# pattern
		'spread': True,
		'phase': 'abstraction',
		'generation': {
			"instruct_task": f'\n\nImagine you want to explain how to transform a new input without knowing its output yet. Your task is to infer the overall pattern that describes the relation between all input-output pairs.',
			"output_format": {
				'Example_1': {
					'pixel_changes': 'regarding the first example, describe the changes between the input and output pixels, focusing on pattern changes', 
					'object_changes': 'regarding the first example, describe the changes between the objects in the input and output grids, focusing on color, size, coordinates, shape, and object number', 
					'parts_of_interest': 'regarding the transformation from input to output, describe the parts of interest of the input grid (e.g. a pixel pattern or objects) and explain their importance; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
					},
				'Example_2': {...},
				'overall_pattern': {
					'conditions': 'why do pixels or objects change? Search for conditions in the input based on colors, positions, and sizes!',
					'overall_pattern': 'describe the input-output relationship valid for all input-output pairs based only on the input. Specify WHAT changes, WHY it changes, and HOW. Be specific!',
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
		'spread': True,
		'phase': 'abstraction',
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
		'spread': True,
		'phase': 'application',
		'generation': {
        	"instruct_task": f'\n\nMoreover, you are given a new test case with a new input grid. Your task is to transform the test input grid into its test output grid.',
			"output_format": {
                'input_description': 'describe the test input grid and identify all objects and pixel pattern', # in the input grid by following the format: [Object_ID: {color: \'object color\', coordinates: [[x_1,y_1], [x_2,y_2], ..], size: number of pixels}, ...]',
                'transformation': 'apply the transformation steps to the test input grid, detailing how each condition of the transformation rules applies to the current task and respond to every step in detail.',
                'transformation_result': 'describe the resulting pixel pattern or objects in the test output grid.',
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

# objects
# prompt_modules = {
#  	"0": {
# 		# pattern
# 		'spread': True,
# 		'phase': 'abstraction',
# 		'generation': {
# 			"instruct_task": f'\n\nImagine you want to explain how to transform a new input without knowing its output yet. Your task is to infer the overall pattern that describes the relation between all input-output pairs.',
# 			"output_format": {
# 				'Example_1': {
# 					'object_number': 'regarding the first example, analyze if and how the number of objects changed from input to output',
# 					'object_analysis': 'regarding the first example, describe the differences between the input and output objects, be precise and say WHAT changed HOW, focus on color, coordinates, size',
# 					'conditions': 'regarding the first example, why do certain objects change? Search for conditions in the input that determine the changes, focus on object colors, coordinates, and sizes!',					
# 					},
# 				'Example_2': {...},
# 				'overall_pattern': {
#         			'conditions': 'regarding all examples, why do certain objects change? Search for conditions in the inputs that determine the changes, focus on object colors, positions, and sizes!',
# 					'overall_pattern': 'define general rules to transform any input into its output based only on the input. Specify WHAT type of object changes, WHY it changes, and HOW. Be specific!',
# 					},
#                 },
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that describes the relation between the input and output grids of all examples.',
# 			"instruct_task": f'\n\nEvaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples. Be as critical as possible with all details!',
# 			"output_format": {
#                 'Example_1': {
#                     'conditions_analysis': 'Regarding the first example, analyze if the given conditions refer only to the input and are relevant to determine the changes.',
#                     'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern describes the transformation from input to output.',
#                     'precision_analysis': 'Regarding the first example, analyze if the given overall pattern is precise enough to transform a new input to its output.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the given pattern as integer.'
#                     },
#                 'Example_2': {...},
#                 }
#    		 	}
#      	},
# 	"1": {
# 		'spread': False,
# 		'phase': 'abstraction',
# 		'generation': {
# 			"instruct_task": f'\n\nYour task is to give detailed transformation steps that are generally applicable to all examples to transform the input grid into its output grid.',
# 			"output_format": {
# 				'Example_1': {
# 					'conditions': 'regarding the first example, list all relevant conditions regarding the input that determine the transformation, focusing on shape, size, coordinates, values, counts, symmetry',
# 					'transformation': 'regarding the first example, describe the transformation steps needed to transform the input grid into its output grid, focus on conditions. Be specific!',
# 					},
# 				'Example_2': {...},
# 				'transformation_steps': 'create a list of detailed transformation steps that are generally applicable to transform a given input grid into its output grid, focus on conditions. Be specific!',
#    		 		},
# 			},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a list of detailed transformation steps that transform an input grid into its output grid.',
# 			"instruct_task": f'\n\nEvaluate the given transformation steps and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all details!',
# 			"output_format": {
#                 'Example_1': {
#                     'transformation_analysis': 'Regarding the first example, analyze if the transformation steps correctly transform the input grid into its output grid.',
#                     'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the transformation steps as integer.'
#                     },
#                 'Example_2': {...},
#                 },
#    		 	},
#      	},
# 	"2": {
# 		'spread': True,
# 		'phase': 'application',
# 		'generation': {
#         	"instruct_task": f'\n\nNext to a few example input-output pairs, you are given a new test case with a new input grid. Your task is to transform the test input grid into its test output grid.',
# 			"output_format": {
#                	'test_case_input_objects': 'copy the objects of the test case input grid from the task',
#                 'transformation': 'Describe in natural language how the transformed objects should look like in the test output grid, be specific and state new object sizes, coordinates, colors. Objects can not overlap!',
# 				'output': {
# 					'test_case_output_dimension': 'state the dimension of the test case output grid [rows, columns] as list of integers',
# 					'transformed_objects': 'Describe all objects after transformation for the test output grid by following the format in the test case input: : "[Object_ID: {\'color\': \'object color\', \'coordinates\': [[row_1,col_1], [row_2,col_2], ..], \'size\': \'number of pixels\'}, ...]"',
# 					},
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a test input grid and a potential test output grid.',
# 			"instruct_task": f'\n\nEvaluate the given test output grid and analyze if the transformation steps were applied correctly to the test input grid. Be as critical as possible with all details!',
# 			"output_format": {
#                 'test_output_analysis': 'consider each transformation step and analyze if the test input grid was correctly transformed into its test output grid.',
#                 'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
#                 }
#    		 	}
#      	},
#    }



################## Prompt modules Single CoT ################
#normal
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

# for object detection
# prompt_modules_naive = {
# 	"0": {
# 		'generation': {
# 			"instruct_task": f'\n\nYou are to infer the relation between inputs and outputs from the examples. Then, your task is to transform the test input grid into its test output grid.',
# 			"output_format": {
# 				'example_1_description': {
# 					'object_number_changes': 'regarding the first example, does the number of objects change from input to output?',
# 					'object_changes': 'regarding the first example, describe the changes between the input and output objects, focusing on color, size, coordinates ([row_index, column_index]), and shape', 
# 					},
# 				'example_2_description': {...},
#     			'overall_pattern': 'describe the input-output relationship valid for all input-output pairs', 
# 				'instructions': 'describe the required transformation actions to transform a new input into its output, think step by step', 
# 				'test_case_input_objects': 'copy the objects of the test case input grid from the task',
# 				'transformation': 'Describe in natural language how the transformed objects should look like in the test output grid, focusing on size, coordinates, color',
# 				'transformed_objects': 'Describe all objects after transformation for the test output sequence by following the format in the test case input.',
# 				'test_case_output_dimension': 'state the dimension of the test case output sequence [rows, columns] as list of integers',
# 				'test_case_output': 'Create the test case output pixel grid with the transformed objects as numpy array, e.g. \"[[0, 0, ...], [...]]\". Use zero-indexing [row_index, column_index] for the object positions and fill unoccupied cells with the background color!'
#              	},
#    		 	},
# 		},
# 	}


################## Few-Shot Examples ################

few_shot_ex = None
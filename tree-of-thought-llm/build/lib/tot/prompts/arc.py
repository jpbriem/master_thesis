################## General Task Explanation ##################
general_explanation = '''You are given a series of example input and output pairs that share the same logical pattern of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'i' represent different colors and '.' represents the background. For example, [['.','b','.'],['.','.','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.

The logical pattern might refer to concepts as follows:
- Goal-directedness: input is start and output is end state of process 
- Geometry & topology:
	- Lines, rectangular shapes.
	- Symmetries, mirroring, rotations, translations.
	- Shape upscaling or downscaling, elastic distortions.
	- Containing / being contained / being inside or outside of a perimeter.
	- Drawing lines, connecting points, orthogonal projections.
	- Copying, repeating.
	- Patterns or mosaic based on sections.
- Objects:
	- Objects are shapes based on similar colors or based on surroundings.
	- Object transformations based on geometry and topology.
	- Touching objects have contact with each other.
	- Noise pixels.
-  Arithmetics based on objects or shapes pixels:
	- Counting.
	- Sorting.

The list is not exhaustive. Transformations can be conditional.'''

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

failure_analysis_prompt = {
    "system": '''You are confronted with a task in which a 2-dimensional grid of pixels should be transformed. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'i' represent different colors, and '.' represents the background. For example, [['.','b','.'],['.','.','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.\n
You are given an input grid and 2 output grids, one is wrong and the other is the ground truth. Your task is to compare the output sequences.\nYou are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.\n''',
	"user": '''Input: {test_input}\nOutput ground truth: {output_gt}\nOutput wrong: {output_wrong}'''
 }

revision_prompt = {
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
		# grid dimension
		'generation': {
			"instruct_task": f'\n\nYour task is to give a description about the dimensions of an input and output grid.',
			"output_format": {
				'Example_1': {
					'grid_view': 'describe the dimension of the input and output grid.', 
					'pixel_view': 'describe the pixels of the input and output grid, focusing on shape, amount, size, position, values, counts, symmetry', 
					'object_view': 'describe the objects in the input grid and output grid, focusing on shape, amount, size, position, values, cell count', 
				},
				'Example_2': {...},
    			'description': {	
     				'input_dimension': 'Regarding all input grids, is there a pattern describing the typical dimension of an input grid? e.g. all have the same fixed dimension or random or varying dimensions with a pattern..',
					'output_dimension': 'Regarding all output grids, is there a pattern describing the typical dimension of an output grid? e.g. all have the same fixed dimension or varying dimensions with a pattern dependend on certain characteristics of the input grid; be specific and explain how to determine the output grid dimension!',
					},
    			},
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a description about the dimensions of an input and output grid.',
			"instruct_task": f'\n\nEvaluate the given description and analyze if it correctly describes the provided input and output grids of all examples. Be as critical as possible with all pattern details!',
			"output_format": {
                'Example_1': {
                    'input_dimension_analysis': 'Regarding the first example, analyze if the input grid fits to the given description.',
                    'output_dimension_analysis': 'Regarding the first example analyze both: 1. does the output grid fit to the given description? 2. is the description helpful to determine the output grid dimension solely based on the given description and the input grid?',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the description as integer.'
                    },
                'Example_2': {...},
                },
     		},
		},
 	"1": {
		# pattern 
		'generation': {
			"instruct_task": f'\n\nYour task is to infer an overall pattern that describes the simplest relation between all input and output pairs.',
			"output_format": {
				'Example_1': {
					'pixel_changes': 'describe the changes between the input and output pixels, focusing on color, coordinates, patterns, counts', 
					'object_changes': 'describe the changes between the objects in the input and output grids, focusing on shape, amount, size, position, values, counts, symmetry', 
					'parts_of_interest': 'regarding the transformation from input to output, describe and explain the importance of the parts of interest of the input grid, e.g. the grid dimension, pixel pattern, or objects; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
					},
				'Example_2': {...},
				'overall_pattern': {
					'parts_of_interest': 'Regarding all examples, is there a pattern about how to determine the parts of interest of the input grid?',
					'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
					},
                },
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given hints to determine parts of interest of the input grid and an overall pattern that describes the relation between the input and output grids of all examples.',
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
	"2": {
		'generation': {
			"instruct_task": f'\n\nYour task is to give step-by-step instructions that are general applicable to all examples to transform the input grid into its output grid.',
			"output_format": {
				'Example_1': {
					'part_of_interest': 'describe the parts of interest of the input grid; be specific and describe the parts appropriately (position, color, shape, size, count, symmetry, etc.)',
					'transformation': 'describe the transformation from input to output step-by-step and refer to the parts of interest',
					'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
					},
				'Example_2': {...},
				'instructions': 'summarize the example transformations and provide step-by-step instructions that are generally applicable to transform an input grid into its output grid, focus on potential transformation conditions and how to solve them', 
   		 		},
			},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given step-by-step instructions that are generally applicable to transform an input grid into its output grid.',
			"instruct_task": f'\n\nEvaluate the given instructions and analyze if they correctly describe the transformation for all examples. Be as critical as possible with all pattern details!',
			"output_format": {
                'Example_1': {
                    'instruction_analysis': 'Regarding the first example, apply the given instructions to the input grid and analyze if they correctly transform the input grid into its output grid.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
                    },
                'Example_2': {
                    'instruction_analysis': '...',
                    'value': '...'
                    },
                },
   		 	},
     	},
	"3": {
		'generation': {
			"instruct_task": f'\n\nMoreover, you are given a new test case with an input grid. Your task is to transform the test input grid into its test output grid.',
			"output_format": {
                'description': 'describe the test input with all its parts of interest and try to determine the dimension of the output grid',
                'intermediate_results': 'apply the instructions step-by-step to the test input grid; focus on potential transformation conditions and provide all intermediate grids',
                'output': 'return only the resulting test output grid as numpy array' 
                }
   		 	},
		'evaluation': {
			"instruct_previous_thoughts": f'\nMoreover, you are given a test input grid and a potential test output grid.',
			"instruct_task": f'\n\nEvaluate the given test output grid and analyze if it fits to the given description, overall pattern, and instructions. Be as critical as possible with all pattern details!',
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
# 			"instruct_task": f'\n\nYour task is to give only an abstract description about how an input grid and how an output grid typically look like based on the examples.',
# 			"output_format": {
#                 'grid_view': 'describe the dimensions of all input grids and of all output grids one after another.', 
#                 'pixel_view': 'describe the pixels of all input grids and of all output grids one after another, focusing on positions, patterns, or counts', 
#                 'object_view': 'describe the objects in all input grids and in all output grids one after another, focusing on shape, amount, size, position, values, cell count', 
# 				'description': {	
#      				'input_description': 'Regarding all input grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical input grid has a dimension of ... and looks like..."',
# 					'output_description': 'Regarding all output grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical output grid has a dimension of ... and looks like..."',
# 					},
#     			},
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.',
# 			"instruct_task": f'\n\nEvaluate the given description and analyze if it correctly describes the provided input and output grids of all examples.',
# 			"output_format": {
#                 'Example_1': {
#                     'description_analysis': 'Regarding the first example, analyze if both the input and output grid fit to the given description.',
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
#                 'grid_changes': 'For each example: describe if and how the dimension of the input grids is different from its output grid', 
#                 'pixel_changes': 'For each example: describe the changes between the input and output pixels, focusing on movement or pattern changes', 
#                 'object_changes': 'For each example: describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
#                 'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given an overall pattern that might describe the relation between the input and output grids of all examples.',
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
# 			"instruct_task": f'\n\nYour task is to give step-by-step instructions that are general applicable to all examples to get from the input grid to its output grid.',
# 			"output_format": {
#                 'part_of_interest': 'regarding the transformation, describe the parts of interest of the input grid, e.g. the grid dimension, pixel pattern, or objects',
#                 'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
#                 'instructions': 'describe all transformation steps with potential conditions and provide step-by-step instructions that are general applicable to transform an input grid into its output grid', 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given step-by-step instructions that are generally applicable to transform an input grid into its output grid.',
# 			"instruct_task": f'\n\nEvaluate the given set of instructions and analyze if it correctly describes the transformation for all examples.',
# 			"output_format": {
#                 'Example_1': {
#                     'instruction_analysis': 'Regarding the first example, analyze if the given instructions correctly transform the input grid into its output grid. ',
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
# 			"instruct_task": f'\n\nMoreover, you are given a new test case with an input grid. Your task is to transform the test input grid into its test output grid.',
# 			"output_format": {
#                 'description': 'describe the test input and check if it fits to the given abstract description',
#                 'intermediate_results': 'apply the instructions step-by-step to the test input grid; focus on potential transformation conditions and provide all intermediate grids',
#                 'output': 'return only the resulting test output grid as numpy array' 
#                 }
#    		 	},
# 		'evaluation': {
# 			"instruct_previous_thoughts": f'\nMoreover, you are given a test input grid and a potential test output grid.',
# 			"instruct_task": f'\n\nEvaluate the given test output grid and analyze if it fits to the given description, overall pattern, and instructions.',
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
			"instruct_task": f'\n\nYou are to infer the simplest possible relation beetween input and output. Then, your task is to transform the test input grid into its test output grid.',
			"output_format": {
          		'grid_changes': 'describe if the dimension of the input grid is different to its output grid', 
				'pixel_changes': 'describe the changes between the input and output pixels, focusing on movement or pattern changes', 
				'object_changes': 'describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
				'overall_pattern': 'describe the simplest input-output relationship for all input-output pairs', 
				'instructions': 'describe the transformation actions in detail step by step', 
				'test_output': 'Use the instructions to transform the test input grid and return only the resulting output grid in numpy array format.'
            	},
   		 	},
		},
	}



few_shot_ex = None
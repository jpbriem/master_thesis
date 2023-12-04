################## prompt modules ##################
general_explanation = '''You are given a series of example input and output pairs that share the same logical pattern of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'j' represent different colors, where 'a' represents the background. For example, [['a','b','a'],['a','a','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.

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

################## Prompts ##################

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

prompt_modules = {
	"0": {
		'generation': {
			"instruct": f'\nYour task is to give only an abstract description about how an input grid and how an output grid typically look like based on the examples.\n',
			"output_format": {
                'grid_view': 'describe the dimensions of all input grids and of all output grids one after another.', 
                'pixel_view': 'describe the pixels of all input grids and of all output grids one after another, focusing on positions or patterns', 
                'object_view': 'describe the objects in all input grids and in all output grids one after another, focusing on shape, amount, size, position, values, cell count', 
				'description': {	
     				'input_description': 'Regarding all input grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical input grid has a dimension of ... and looks like...',
					'output_description': 'Regarding all output grids, summarize your findings about the dimensions, pixel view and object view in an abstract description by completing the sentence: "A typical output grid has a dimension of ... and looks like...',
					},
    			},
   		 	},
		'evaluation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Evaluate the given description and analyze if it correctly describes the provided input and output grids of all examples.\n''',
			"output_format": {
                'Example_1': {
                    'description_analysis': 'Regarding the first example, analyze if both the input and output grid fit to the given description.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the description as integer.'
                    },
                'Example_2': {
                    'description_analysis': '...',
                    'value': '...'
                    },
                }
   		 	}
     	},
 	"1": {
		'generation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Your task is to infer an overall pattern that describes the simplest relation between all input and output pairs.\n''',
			"output_format": {
                'grid_changes': 'For each example: describe if and how the dimension of the input grids is different from its output grid', 
                'pixel_changes': 'For each example: describe the changes between the input and output pixels, focusing on movement or pattern changes', 
                'object_changes': 'For each example: describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count', 
                'overall_pattern': 'summarize your findings and describe the simplest input-output relationship valid for all examples', 
                }
   		 	},
		'evaluation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
Evaluate the given pattern and analyze if it correctly describes the relation between the inputs and outputs of all examples.\n''',
			"output_format": {
                'Example_1': {
                    'overall_pattern_analysis': 'Regarding the first example, analyze if the given overall pattern correctly describes the relation between the input and output.',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the overall pattern as integer.'
                    },
                'Example_2': {
                    'overall_pattern_analysis': '...',
                    'value': '...'
                    },
                }
   		 	}
     	},
	"2": {
		'generation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between the input and output grids of all examples.
Your task is to give step-by-step instructions that are general applicable to all examples to get from the input grid to its output grid.\n''',
			"output_format": {
                'part_of_interest': 'regarding the transformation, describe the parts of interest of the input grid, e.g. the grid dimension, pixel pattern, or objects',
                'conditions': 'describe if and how the transformation process is based on conditions, e.g. object characteristics (number, shape, symmetry, color, size, position) or pixel characteristics (color, position)',
                'instructions': 'describe all transformation steps with potential conditions and provide step-by-step instructions that are general applicable to transform an input grid into its output grid', 
                }
   		 	},
		'evaluation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
Moreover, you are given a set of instructions that might be generally applicable to transform an input grid into its output grid.
Evaluate the given set of instructions and analyze if it correctly describes the transformation for all examples.\n''',
			"output_format": {
                'Example_1': {
                    'instruction_analysis': 'Regarding the first example, analyze if the given instructions correctly transform the input grid into its output grid. ',
                    'value': 'Based on your analysis regarding the first example, give a rating between 0 and 10 for the instructions as integer.'
                    },
                'Example_2': {
                    'instruction_analysis': '...',
                    'value': '...'
                    },
                }
   		 	}
     	},
	"3": {
		'generation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between the input and output grids of all examples.
Moreover, you are given step-by-step instructions that are general applicable to transform an input grid into its output grid.
Based on the provided information, your task is to apply the general instructions to a new test case and you are to transform the test input grid into its test output grid.\n''',
			"output_format": {
                'description': 'describe the test input and check if it fits to the given abstract description',
                'intermediate_results': 'apply the instructions step-by-step to the test input grid; focus on potential transformation conditions and provide all intermediate grids',
                'test_output': 'return only the resulting test output grid as numpy array' 
                }
   		 	},
		'evaluation': {
			"instruct": '''\nMoreover, you are given an abstract description about how an input grid and an output grid typically look like.
Moreover, you are given an overall pattern that might describe the relation between all input and output pairs.
Moreover, you are given a set of instructions that might be generally applicable to transform an input grid into its output grid.
Moreover, you are given a test input grid and a potential test output grid.
Evaluate the given test output grid and analyze if it fits to the given description, overall pattern, and instructions.\n''',
			"output_format": {
                'test_output_analysis': 'analyze if the given test output fits to the given description, overall pattern, and instructions.',
                'value': 'Based on your analysis, give a rating between 0 and 10 for the test output as integer.'
                }
   		 	}
     	},
   }
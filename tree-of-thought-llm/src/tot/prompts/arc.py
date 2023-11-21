################## prompt modules ##################
general_explanation = '''You are given a series of inputs and output pairs that share the same logic of getting the output from its input. Each input and output is a 2-dimensional grid of pixels. The values from 'a' to 'j' represent different colors, where 'a' represents the background. For example, [['a','b','a'],['a','a','c']] represents a 2 row x 3 column grid with color 'b' at position (1,0) and color 'c' at position (2,1). The coordinates are 2D coordinates (row, column), row representing row number, column representing col number, with zero-indexing.
You are to infer the simplest possible relation beetween input and output. The given sample pairs may not reflect all possibilities.

You can refer to concepts as follows:
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

standard_prompt = '''{context}{input}'''

cot_prompt = general_explanation + '''

{special_instructions}

You are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.

{context}{input}'''


vote_prompt = general_explanation + '''
{special_instructions}
You are to output only the following in json format: {output}. Do not use quotation marks ' or " within the fields.

{context}{input}'''

compare_prompt = '''

'''

score_prompt = '''

'''
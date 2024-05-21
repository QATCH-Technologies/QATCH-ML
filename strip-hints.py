def strip_hints(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Remove comments and blank lines from each line
    cleaned_lines = []
    for line in lines:
        # Remove inline comments (if any)
        line = line.split('#', 1)[0]
        cleaned_line = line.rstrip()  # Remove leading/trailing whitespace
        if cleaned_line:  # Only add non-blank lines
            cleaned_lines.append(cleaned_line)

    with open(output_file, 'w') as outfile:
        outfile.write('\n'.join(cleaned_lines))

# Usage: replace 'input_file.py' and 'output_file.py' with actual filenames
strip_hints('qatch_identify_points_with_maths.py', 'ModelData.py')

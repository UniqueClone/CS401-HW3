def format(s1):
    f = open('testing.txt')

    headings = ""
    body = []
    current_section = ""
    current_heading = ""

    # Read each line of file individually
    for line in f:
        # Check for header lines, starting with "=="
        if line[0:2] == "==":
            # Add headings to lists, will be first line of output
            if line[2] == ' ': # Remove space before heading
                headings += (line[3:-1] + "|") # Save heading
            else:
                headings += (line[2:-1] + "|") # Save heading
            # After finding new section title, if current section 
            # is not empty, save current section to body
            if current_section != "":
                print(current_section.replace("\n", ""))
                body.append(current_section.replace("\n", ""))
                current_section = ""

        # If line is not a section heading, empty or a blank new line, 
        # add to current_section
        elif (line != "") and (line != "\n"):
            current_section += line
        elif (line == "") or (line == "\n"):
            continue
        # Else print so we know what has been missed
        else:
            print("Missed line: ", line)
    print(headings)

    return headings, body

def output(headings_string, body_array):

    # Write headings and body to file
    # This will hopefully be comma separated data
    f2 = open('body.txt', 'w')
    f2.write(headings_string + "\n")
    f2.write("|".join(body_array)[0:-1])

# Input File
headings, body = format('testing.txt')
output(headings, body)




# Headings in file:
# == SOURCE
# == AGENT
# == GOAL
# == DATA
# == METHODS
# == RESULTS
# == ISSUES
# == SCORE
# == COMMENTS
# This file contains all the text to populate the main page

help_text = '''
            # Help
            ## New in v0.0.3a:\n
                1. Modularised code for ease of understanding and usage
                2. The most awaited "Update" button for the DV and QoI sliders
                3. Add images like ADGs or system diagrams by dragging and dropping them into the tool
                4. 3D potting of solution spaces
                5. Introducing probing `Nominal` design by collapsing the design space, the nominal of the sliders
                6. Sample design from the design space (for custom visualisation or processing)
                7. All the features from the previous versions are carried on
                8. Two examples: Line and Crash Design are included as templates for setting up problems.
                Users are encouraged to use the template Line from the current version to set up new problems.
                9. A black square now highlights the nominal design
                10. Progress update tracker of joblib in terminal using tqdm


            ## Using the X-Ray tool:\n
                1. Configure the problem from the problem configuration expander on the sidebar\n
                    - "Select problem" dropdown populates all the available problems in the "problems" folder\n
                    - Each problem consists of an input, output and library directories\n
                    - The input directory has all the initial design space boundaries of all the design variables and the quantities of interest
                      in the "dv_space.csv" and "qoi_space.csv" files respectively\n
                    - The library directory contains all the routines corresponding to the bottom-up mappings required for the evaluation of the sampled design variables\n
                    - All the results, including the final plots and the solution spaces are saved in the output folder\n
                    - Once the problem is selected from the dropdown, the entire design space is sampled and the result is visualised in the output plots upon clicking `Rerun`\n
                    - The sample size can be modified easily through the sample size text box
                2. Modify the plot configuration as required\n
                    - The opacity slider allows one to change the opacity of all the constraints\n
                    - Marker size varies the size of each point used in the plots\n
                    - Based on the screen used, the number of plots per column can also be modified using the "Columns" field\n
                3. The list of design variables is automatically read from the csv file and their corresponding sliders are produced
                   in the "Design variables" dropdown to manipulate the respective variable bounds\n
                4. Similarly all the sliders corresponding to the quantities of interest are also generated automatically\n
                5. The "Export" expander containts options to:\n
                    - Save the generated plots as .pdf format in the root directory\n
                    - (WIP) Automatically generate the Attribute Dependency Graph (ADG) of the selected problem and export it as a .pdf file \n

            ## Remarks
                1. The scale of the DVs and QoI should be appropriately chosen, for example a very small DV will always be shown as 0 in the sliders
                One way to solve this problem is to multiply the variables with a constant factor to bring them to good scale (order of 10s-100s)

            ## Setting up a problem in the X-Ray tool\n
            Note: The names of the qoi and the functions defined in the problem class should be the same and the return variables should also be names same as the qois given in the input excel sheet, qoi_space\n
            ### Example
            #### Problem name: __line__\n
            #### Problem description:
                1. The problem contains three design variables x, y and z
                2. All the variables are bound to be between -100 and 100
                3. The quantities of interest (QoI) are L1 = x-y and L2 = z-y
                4. The QoI are bound between -50 and 50
                5. The problem is to identify the largest bound on x, y and z such that all combinations of these variables within those bounds satisfy the bounds on the QoI
            #### Explore design space:
                1. Valid designs (sampled x, y and z) are plotted in green
                2. Move the sliders under the "Design variables" expander until your design space contains only feasible designs (green dots)
            #### Visualise the solution:
                1. If you do not want to try it yourself, you can load the existing solution by selecting the "Load solution" check box
            '''

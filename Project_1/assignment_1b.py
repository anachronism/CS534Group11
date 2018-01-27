# Urban Planning Problem: Determine ideal location of industry, 
#							commerce, residential sections.
# Inputs: Map with: X at former toxic waste site. 
#						Industrial zones within 2 tiles get a -10.
#						Commercial and residential zones within 2 tiles get -20.
#						Cannot build directly on top of X.
#					S with scenic view.			
#						REsidential zones within 2 tiles get + 10.
#						Can be built on, but removes benefit.
#					0...9:
#						Cost to build on square. 
# Actions: Place tiles.
#				For each industrial tile within 2, there is a +3 bonus.
#				For each residential tile within 3 of a commercial tile, +5.
#				For each commercial tile within 2 squares of a commercial tile, -5.
#				For each industrial within 3 of a residential tile, -5.
#				For each commercial within 3 of a residential, +5.
# 			Use Manhattan distance.
# Algorithms:hill climbing with restarts, genetic algorithms.

# Program behavior:
# Input is a file, where first 3 lines are numbers of industrial, commercial, residential locs.
# Following is a rectangular map of terrain to place the town.
# Run for ~ 10 seconds.
# Output to file (map score, time score was first achieved, a marked map).
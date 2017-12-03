data seismic; 
infile "/home/ddyer0/sasuser.v94/Stats2/Data/seismic-bumps.csv" firstobs=1 dlm=',';
input seismic $ semiacoustic $ shift $ genergy gpuls gdenergy gdpuls ghazard $ nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 energy maxenergy class $;
run;

proc logistic data = seismic plots=(ROC Effect INFLUENCE(UNPACK));
class seismic semiacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89;
model class = seismic semiacoustic shift genergy gpuls gdenergy gdpuls ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89;
run;

* all the interactions to l2;
proc logistic data = seismic plots=(ROC Effect INFLUENCE(UNPACK));
class seismic semiacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89;
model class = seismic | semiacoustic | shift | genergy | gpuls | gdenergy | gdpuls | ghazard | nbumps | nbumps2 | nbumps3 | nbumps4 | nbumps5 | nbumps6 | nbumps7 | nbumps89 @2;
run;

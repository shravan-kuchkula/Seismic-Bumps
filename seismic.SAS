
data seismic; 
infile "/home/ddyer0/sasuser.v94/Stats2/Data/seismic-bumps.csv" firstobs=1 dlm=',';
input seismic $ seismoacoustic $ shift $ genergy gpuls gdenergy gdpuls ghazard $ nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 energy maxenergy class $;
run;

proc logistic data = seismic plots=(ROC Effect INFLUENCE(UNPACK));
class seismic seismoacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89;
model class = seismic semiacoustic shift genergy gpuls gdenergy gdpuls ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89;
run;

proc logistic data = seismic plots=(ROC INFLUENCE);
class seismic seismoacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 ;
model class (event='1') = seismic semiacoustic shift genergy gpuls gdenergy gdpuls ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 / scale = none aggregate lackfit influence;
run;

* main effects model;
proc logistic data=seismic ;
class seismic seismoacoustic shift ghazard / param=ref;
model class(event='1')= seismic seismoacoustic shift ghazard/ scale=none aggregate lackfit influence;
effectplot;
effectplot slicefit(sliceby=shift) / noobs;
run;

* main effects forward selection model;
proc logistic data=seismic ;
class seismic seismoacoustic shift ghazard / param=ref;
model class(event='1')= seismic seismoacoustic shift ghazard/ selection = FORWARD start=3 scale=none details lackfit influence;
effectplot;
effectplot slicefit(sliceby=shift) / noobs;
run;


* all the things model;
proc logistic data=seismic ;
class seismic seismoacoustic shift ghazard nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89/ param=ref;
model class(event='1')= seismic seismoacoustic shift ghazard nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89/ scale=none aggregate lackfit influence;
effectplot;
effectplot slicefit(sliceby=shift) / noobs;
run;

* all the things forward selection;
proc logistic data=seismic ;
class seismic seismoacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 / param=ref;
model class(event='1')= seismic seismoacoustic shift ghazard nbumps nbumps2 nbumps3 nbumps4 nbumps5 nbumps6 nbumps7 nbumps89 / selection = FORWARD start=3 scale=none details lackfit influence;
effectplot;
effectplot slicefit(sliceby=shift) / noobs;
run;
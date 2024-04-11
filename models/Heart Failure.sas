/* Generated Code (IMPORT) */
/* Source File: S1Data.csv */
/* Source Path: /home/u63141156/MY Folder/Machine Learning */
/* Code generated on: 4/10/24, 10:31 AM */

%web_drop_table(WORK.IMPORT1);


FILENAME REFFILE '/home/u63141156/MY Folder/Machine Learning/S1Data.csv';

PROC IMPORT DATAFILE=REFFILE
	DBMS=CSV
	OUT=WORK.heart1;
	GETNAMES=YES;
RUN;

PROC CONTENTS DATA=WORK.heart1; RUN;


%web_open_table(WORK.heart1);


proc means data = WORK.heart1;
var time event gender smoking;
run;

proc univariate data = work.heart1;
var time; 
run;

* nonparametric models
* Graph of Surivival Functions;
proc lifetest data = work.heart1 plots =(s) notable;
time time*event(0);
run;

* Grpahs of survival and hazard functions; 
proc lifetest data = work.heart1 method = act plots = (s(name=Actsurv), h(name=Acthaz) ) notable;
time time*event(0);
run;

* Survival curves by group and univariate test;
proc lifetest data = work.heart1 plots = (s) notable;
time time*event(0);
*strata gender;
strata gender diabetes smoking;
run;

*paramentric and semiparamtric models;

proc phreg data = work.heart;
model time*event(0) = Age Ejection.Fraction Pletelets;
run; 


























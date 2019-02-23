@set Input_qaPariFile=.\evaltool\sample.QApair.txt
@set Input_resultFile=.\evaltool\sample.score.txt
@set Output_metricResultFile=.\evaltool\sample.result.txt

call .\evaltool\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%


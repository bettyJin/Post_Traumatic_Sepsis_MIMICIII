#import gzip #local(Drive) mimic dataset( file type .csv.gz file)
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime


####################### Demographics ##################################################################
# ------------------------------------------------------------------
# Description: This query provides a useful set of information regarding patient ICU stays. 
#              The information is combined from the admissions, patients, and icustays tables. 
#              It includes age, length of stay, sequence, and expiry flags, etc.
# MIMIC version: MIMIC-III v1.4
# A modify veristion of: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/demographics/icustay_detail.sql
# modified list:
#   1. duration result is expressed in hours & in days
# ------------------------------------------------------------------
# this table includes:
# *   Patient Info:
#  >  'subject_id', 'gender', 'dod', 'ethnicity','ethnicity_grouped',
# *   Hosipital Admission Info:
#  >  'hadm_id', 'admittime', 'dischtime', 'los_hospital_hours', 'admission_age' 'hospital_expire_flag'(0/1), 'hospstay_seq', 'first_hosp_stay'(T/F)
# * ICUstay Info:
#  > 'icustay_id', 'intime', 'outtime', 'los_icu_hours', 'icustay_seq', 'first_icu_stay'(T/F)

def demog_sql2df(project_id, saved_path=None):
  demog_query = """
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
  -- patient level factors
  , pat.gender, pat.dod
  -- hospital level factors
  , adm.admittime, adm.dischtime
  , DATETIME_DIFF(adm.dischtime, adm.admittime, DAY) as los_hospital_days
  , DATETIME_DIFF(adm.dischtime, adm.admittime, HOUR) as los_hospital_hours 
  , DATETIME_DIFF(ie.intime, pat.dob, YEAR) as admission_age
  , adm.ethnicity
  , case when ethnicity in
    (
        'WHITE' --  40996
      , 'WHITE - RUSSIAN' --    164
      , 'WHITE - OTHER EUROPEAN' --     81
      , 'WHITE - BRAZILIAN' --     59
      , 'WHITE - EASTERN EUROPEAN' --     25
    ) then 'white'
    when ethnicity in
    (
        'BLACK/AFRICAN AMERICAN' --   5440
      , 'BLACK/CAPE VERDEAN' --    200
      , 'BLACK/HAITIAN' --    101
      , 'BLACK/AFRICAN' --     44
      , 'CARIBBEAN ISLAND' --      9
    ) then 'black'
    when ethnicity in
      (
        'HISPANIC OR LATINO' --   1696
      , 'HISPANIC/LATINO - PUERTO RICAN' --    232
      , 'HISPANIC/LATINO - DOMINICAN' --     78
      , 'HISPANIC/LATINO - GUATEMALAN' --     40
      , 'HISPANIC/LATINO - CUBAN' --     24
      , 'HISPANIC/LATINO - SALVADORAN' --     19
      , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
      , 'HISPANIC/LATINO - MEXICAN' --     13
      , 'HISPANIC/LATINO - COLOMBIAN' --      9
      , 'HISPANIC/LATINO - HONDURAN' --      4
    ) then 'hispanic'
    when ethnicity in
    (
        'ASIAN' --   1509
      , 'ASIAN - CHINESE' --    277
      , 'ASIAN - ASIAN INDIAN' --     85
      , 'ASIAN - VIETNAMESE' --     53
      , 'ASIAN - FILIPINO' --     25
      , 'ASIAN - CAMBODIAN' --     17
      , 'ASIAN - OTHER' --     17
      , 'ASIAN - KOREAN' --     13
      , 'ASIAN - JAPANESE' --      7
      , 'ASIAN - THAI' --      4
    ) then 'asian'
    when ethnicity in
    (
        'AMERICAN INDIAN/ALASKA NATIVE' --     51
      , 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE' --      3
    ) then 'native'
    when ethnicity in
    (
        'UNKNOWN/NOT SPECIFIED' --   4523
      , 'UNABLE TO OBTAIN' --    814
      , 'PATIENT DECLINED TO ANSWER' --    559
    ) then 'unknown'
    else 'other' end as ethnicity_grouped
    -- , 'OTHER' --   1512
    -- , 'MULTI RACE ETHNICITY' --    130
    -- , 'PORTUGUESE' --     61
    -- , 'MIDDLE EASTERN' --     43
    -- , 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' --     18
    -- , 'SOUTH AMERICAN' --      8
  , adm.hospital_expire_flag
  , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN True
      ELSE False END AS first_hosp_stay
  -- icu level factors
  , ie.intime, ie.outtime
  , DATETIME_DIFF(ie.outtime, ie.intime, DAY) as los_icu_days
  , DATETIME_DIFF(ie.outtime, ie.intime, HOUR) as los_icu_hours
  , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
  -- first ICU stay *for the current hospitalization*
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN True
      ELSE False END AS first_icu_stay
  FROM `physionet-data.mimiciii_clinical.icustays` ie
  INNER JOIN `physionet-data.mimiciii_clinical.admissions` adm
      ON ie.hadm_id = adm.hadm_id
  INNER JOIN `physionet-data.mimiciii_clinical.patients` pat
      ON ie.subject_id = pat.subject_id
  WHERE adm.has_chartevents_data = 1
  ORDER BY ie.subject_id, adm.admittime, ie.intime;
  """
  demog_df = run_query(demog_query, project_id)
  if saved_path != None:
    demog_df.to_csv(os.path.join(saved_path, "demographics.csv"))
  return demog_df


# def vital_signs_sql2df(project_id, saved_path=None):
#   vs_query = """
#   -- This query pivots the vital signs for the first 24 hours of a patient's stay
#   -- Vital signs include heart rate, blood pressure, respiration rate, and temperature

#   with ce as
#   (
#     select ce.hadm_id, ce.icustay_id
#       , ce.charttime
#       , (case when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then valuenum else null end) as heartrate
#       , (case when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then valuenum else null end) as sysbp
#       , (case when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then valuenum else null end) as diasbp
#       , (case when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then valuenum else null end) as meanbp
#       , (case when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then valuenum else null end) as resprate
#       , (case when itemid in (223761,678) and valuenum > 70 and valuenum < 120 then (valuenum-32)/1.8 -- converted to degC in valuenum call
#                 when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then valuenum else null end) as tempc
#       , (case when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then valuenum else null end) as spo2
#       , (case when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then valuenum else null end) as glucose
#     FROM `physionet-data.mimiciii_clinical.chartevents` ce
#     -- exclude rows marked as error
#     where (ce.error IS NULL OR ce.error != 1)
#     and ce.icustay_id IS NOT NULL
#     and ce.itemid in
#     (
#     -- HEART RATE
#     211, --"Heart Rate"
#     220045, --"Heart Rate"

#     -- Systolic/diastolic

#     51, --	Arterial BP [Systolic]
#     442, --	Manual BP [Systolic]
#     455, --	NBP [Systolic]
#     6701, --	Arterial BP #2 [Systolic]
#     220179, --	Non Invasive Blood Pressure systolic
#     220050, --	Arterial Blood Pressure systolic

#     8368, --	Arterial BP [Diastolic]
#     8440, --	Manual BP [Diastolic]
#     8441, --	NBP [Diastolic]
#     8555, --	Arterial BP #2 [Diastolic]
#     220180, --	Non Invasive Blood Pressure diastolic
#     220051, --	Arterial Blood Pressure diastolic


#     -- MEAN ARTERIAL PRESSURE
#     456, --"NBP Mean"
#     52, --"Arterial BP Mean"
#     6702, --	Arterial BP Mean #2
#     443, --	Manual BP Mean(calc)
#     220052, --"Arterial Blood Pressure mean"
#     220181, --"Non Invasive Blood Pressure mean"
#     225312, --"ART BP mean"

#     -- RESPIRATORY RATE
#     618,--	Respiratory Rate
#     615,--	Resp Rate (Total)
#     220210,--	Respiratory Rate
#     224690, --	Respiratory Rate (Total)


#     -- spo2, peripheral
#     646, 220277,

#     -- glucose, both lab and fingerstick
#     807,--	Fingerstick glucose
#     811,--	glucose (70-105)
#     1529,--	glucose
#     3745,--	Bloodglucose
#     3744,--	Blood glucose
#     225664,--	glucose finger stick
#     220621,--	glucose (serum)
#     226537,--	glucose (whole blood)

#     -- TEMPERATURE
#     223762, -- "Temperature Celsius"
#     676,	-- "Temperature C"
#     223761, -- "Temperature Fahrenheit"
#     678 --	"Temperature F"

#     )
#   )
#   select 
#     ce.hadm_id,  ce.icustay_id
#     , ce.charttime
#     , avg(heartrate) as heartrate
#     , avg(sysbp) as sysbp
#     , avg(diasbp) as diasbp
#     , avg(meanbp) as meanbp
#     , avg(resprate) as resprate
#     , avg(tempc) as tempc
#     , avg(spo2) as spo2
#     , avg(glucose) as glucose
#   from ce
#   group by ce.hadm_id, ce.icustay_id, ce.charttime
#   order by ce.hadm_id, ce.icustay_id, ce.charttime
#   ;
#   """
#   vs_df = run_query(vs_query, project_id)
#   if saved_path != None:
#     vs_df.to_csv(os.path.join(saved_path, "pivot_vital.csv"))
#   return vs_df


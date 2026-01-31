create database scamshield;

use scamshield;

select * from scans;

ALTER TABLE scans
ADD COLUMN cluster_id INT NULL,
ADD COLUMN cluster_terms TEXT NULL;

select * from scans;
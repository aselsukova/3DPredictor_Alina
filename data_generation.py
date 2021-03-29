import sys
import os

head_folder_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor"
source_path = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor/source"
source_path2 = os.path.dirname(os.path.abspath(sys.argv[0]))+"/3Dpredictor/nn/source"
sys.path.append(source_path)
sys.path.append(source_path2)

import logging
from ChiPSeqReader import ChiPSeqReader
from Contacts_reader import ContactsReader
from hicFileReader import hicReader
from fastaFileReader import fastaReader
from RNASeqReader import RNAseqReader
from TssReader import TssReader
from E1_Reader import E1Reader,fileName2binsize
from shared import Interval, Parameters
from DataGenerator import generate_data
from PredictorGenerators import E1PredictorGenerator,ChipSeqPredictorGenerator, \
                SmallChipSeqPredictorGenerator,SmallE1PredictorGenerator, \
                SitesOrientPredictorGenerator, OrientBlocksPredictorGenerator, ConvergentPairPredictorGenerator, Distance_to_TSS_PG
from VectPredictorGenerators import loopsPredictorGenerator
from LoopReader import LoopReader
import pandas as pd
import pickle
import datetime
import configparser


file = sys.argv[1]
input = open(file,'r')
lines = input.readlines()
args = {}

#change to configparser
for line in lines:
    line = line.strip().split()
    args[line[0]] = line[1]

logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

if __name__ == '__main__': #Requered for parallization, at least on Windows
    logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)

    input_folder = args['input_folder']
    output_folder = args['output_folder']
    cell_type = args['cell_type']
    start = int(args['start'])
    end = int(args['end'])
    chromosome = 'chr' + args['chr_num']
    hic_name = args['hic_name']
    CTCF_file_name = args['CTCF_file_name']
    #RNA_file_name = args['RNA_file_name']

    # validate_chrs = args['validate_chrs'].split(",")
    # for chr in validate_chrs:
    #     chr = int(chr)

    params = Parameters()
    params.binsize = int(args['binsize']) #sequence resolution of contacts data. Use for finding of normalized coefficient file
    params.window_size = params.binsize #region around contact to be binned for predictors. Usually equal to binsize
    params.mindist = params.binsize*2+1 #minimum distance between contacting regions
    params.maxdist = 1500000
    # params.sample_size = end - start
    params.sample_size = 2 #how many contacts write to file
    #params.conttype = conttype
    params.max_cpus = int(args['max_cpus'])
    params.keep_only_orient = False
    params.use_only_contacts_with_CTCF = "all_cont"#"all_cont" or "cont_with_CTCF"
    rearrangement = False



   # deletion = Interval("chr" + chr_num, start, end)
    write_all_chrms_in_file=False #set True if you want write training file consisting several chromosomes
    fill_empty_contacts = False #set True if you want use all contacts in region, without empty contacts

    logging.getLogger(__name__).debug("Using input folder "+input_folder)

    # Read contacts data
    genome = fastaReader(args['path_to_genome'], useOnlyChromosomes=[chromosome])#str(chr_num)])
    genome = genome.read_data()
    print(genome.data)
    now = datetime.datetime.now()
    params.contacts_reader = hicReader(fname=input_folder +"/"+ cell_type +"/"+ hic_name,
                                       genome=genome,
                                       binsize=params.binsize)
    params.contacts_reader = params.contacts_reader.read_data(fill_empty_contacts=fill_empty_contacts,
                                                              noDump=False)

    if params.use_only_contacts_with_CTCF == "cont_with_CTCF":
        params.proportion = 1
        params.contacts_reader.use_contacts_with_CTCF(CTCFfile=input_folder+"/" + cell_type+"/CTCF/"+CTCF_file_name,
                                                        maxdist=params.maxdist,
                                                        proportion=params.proportion,
                                                        keep_only_orient=params.keep_only_orient,
                                                        CTCForientfile=input_folder + "/" + cell_type + \
                                                                       "/CTCF/"+CTCF_file_name+"-orient.bed")
        params.use_only_contacts_with_CTCF += str(params.contacts_reader.conts_with_ctcf)
        #make deletion
    if rearrangement:
        params.contacts_reader.delete_region(deletion)
        # Read CTCF data
    logging.info('create CTCF_PG')
    params.ctcf_reader = ChiPSeqReader(input_folder + "/"+ cell_type+"/CTCF/"+CTCF_file_name, name="CTCF")
    params.ctcf_reader.read_file()
    #params.ctcf_reader.set_sites_orientation(input_folder +"/"+ cell_type+"/CTCF/"+CTCF_file_name+"-orient.bed")
    if params.keep_only_orient:
        params.ctcf_reader.keep_only_with_orient_data()
    if rearrangement:
        params.ctcf_reader.delete_region(deletion)
    OrientCtcfpg = SitesOrientPredictorGenerator(params.ctcf_reader,N_closest=4)
    NotOrientCTCFpg = SmallChipSeqPredictorGenerator(params.ctcf_reader,params.binsize,N_closest=4)
    ConvergentPairPG = ConvergentPairPredictorGenerator(params.ctcf_reader, binsize=params.binsize)

        # Read CTCF data and drop sites w/o known orientation
    # params.ctcf_reader_orientOnly = ChiPSeqReader(input_folder+"/" + cell_type+"/CTCF/"+CTCF_file_name,name="CTCF")
    # params.ctcf_reader_orientOnly.read_file()
    # params.ctcf_reader_orientOnly.set_sites_orientation(input_folder + "/" + cell_type+"/CTCF/"+CTCF_file_name+"-orient.bed")
    # params.ctcf_reader_orientOnly.keep_only_with_orient_data()
    if rearrangement:
        params.ctcf_reader_orientOnly.delete_region(deletion)

    OrientBlocksCTCFpg = OrientBlocksPredictorGenerator(params.ctcf_reader_orientOnly,params.binsize)

        # #Read other chip-seq data
        # logging.info('create chipPG')
        # chipPG = []
        # filenames_df = pd.read_csv(input_folder + "H1/Chip-seq/filenames.csv")
        # # assert len(os.listdir(input_folder + 'peaks/')) - 1 == len(filenames_df['name'])
        # # print(len(os.listdir(input_folder + 'peaks/')))
        # # print(len(filenames_df['name']))
        # # proteins=set(["RAD21", "SMC3", "POLR2A", "H3K27ac", "H3K27me3", "DNase-seq", "H3K9me3", "H3K4me1", "H3K4me2", "H3K4me3", "YY1"])
        # for index, row in filenames_df.iterrows():
        #     # if row["name"] in proteins:
        #         params.chip_reader = ChiPSeqReader(input_folder + 'H1/Chip-seq/' + row["filename"], name=row['name'])
        #         params.chip_reader.read_file()
        #         chipPG.append(SmallChipSeqPredictorGenerator(params.chip_reader,params.window_size,N_closest=4))

    #     #Read RNA-Seq data
    # params.RNAseqReader = RNAseqReader(fname=input_folder + "/"+cell_type +"/RNA-seq/"+RNA_file_name,
    #                                        name="RNA")
    # params.RNAseqReader.read_file(rename={ "gene_id": "gene",
    #                           "Gene start (bp)": "start",
    #                           "Gene end (bp)": "end",
    #                           "Chromosome/scaffold name": "chr",
    #                           "FPKM": "sigVal"},
    #                   sep="\t")
    # if rearrangement:
    #     params.RNAseqReader.delete_region(deletion)
    # RNAseqPG = SmallChipSeqPredictorGenerator(params.RNAseqReader,window_size=params.binsize, N_closest=3)

    params.pgs = [OrientCtcfpg, NotOrientCTCFpg, OrientBlocksCTCFpg,ConvergentPairPG]#RNAseqPG]#+chipPG#+cagePG+metPG
        # # Generate train
        # train_chrs=[]
        # [train_chrs.append("chr"+chr) for chr in chr_nums]
        # if write_all_chrms_in_file:
        #     train_file_name="training.RandOn"+ str(params)
        #     params.out_file=output_folder+"_".join(train_chrs)+train_file_name
        # for trainChrName in train_chrs:
        #     print(trainChrName)
        #     # training_file_name = "training.RandOn" + trainChrName + str(params) + ".txt"
        #
        #     params.sample_size = len(params.contacts_reader.data[trainChrName])
        #     params.interval = Interval(trainChrName,
        #                           params.contacts_reader.get_min_contact_position(trainChrName),
        #                           params.contacts_reader.get_max_contact_position(trainChrName))
        #
        #     # params.out_file = output_folder + training_file_name
        #     if not write_all_chrms_in_file:
        #         train_file_name = "training.RandOn" + str(params) + ".txt"
        #         params.out_file = output_folder + params.interval.toFileName() + train_file_name
        #     generate_data(params,saveFileDescription=True)
        #     if not write_all_chrms_in_file:
        #         del(params.out_file)
        #     del (params.sample_size)


        # Generate test
    validate_chrs=[] #no need to set chr for validation here!!!!
    [validate_chrs.append("chr"+chr) for chr in chr_nums]#,"chr16", "chr17"]#, "chr18"]#, "chr18", "chr19", "chr20"]#,"chr14", "chr15"]
    if write_all_chrms_in_file:
        validation_file_name = "validatingOrient." + str(params) + ".txt"
        params.out_file = output_folder + "_".join(validate_chrs) + validation_file_name
    for validateChrName in validate_chrs:
        print("chromosome", validateChrName)
        interval=Interval(chromosome, start, end)
        #params.sample_size = len(params.contacts_reader.data[validateChrName])

            # params.interval = Interval(validateChrName,
            #                            params.contacts_reader.get_min_contact_position(validateChrName),
            #                            params.contacts_reader.get_max_contact_position(validateChrName))
        params.interval = interval
        logging.getLogger(__name__).info("Generating validation dataset for interval "+str(params.interval))
        if not write_all_chrms_in_file:
            validation_file_name = "validatingOrient." + str(params) + ".txt"
            params.out_file = output_folder +"/"+ cell_type+params.interval.toFileName() + validation_file_name
        generate_data(params)
        if not write_all_chrms_in_file:
            del(params.out_file)
        del (params.sample_size)

        # for interval in [Interval("chr2", 118000000, 129000000)]:
        # #                  Interval("chr10", 47900000, 53900000),
        # #                  Interval("chr10", 15000000, 20000000),
        # #                  Interval("chr10",36000000,41000000)]:
        # # Interval("chr1", 100000000, 110000000)]:
        #    logging.getLogger(__name__).info("Generating validation dataset for interval "+str(interval))
        #    validation_file_name = "validatingOrient." + str(params) + ".txt"
        #    params.interval = interval
        #    params.out_file = output_folder + params.interval.toFileName() + validation_file_name
        #    generate_data(params)

        # for object in [params.contacts_reader]+params.pgs:
        #     lostInterval = Interval("chr1",103842568,104979840)
        #     object.delete_region(lostInterval)
        #     params.interval = Interval("chr1",100000000,109000000)
        #     logging.getLogger(__name__).info("Saving data to file "+params.interval.toFileName() + "DEL." + lostInterval.toFileName()+validation_file_name)
        # params.out_file = params.interval.toFileName() + "DEL." + lostInterval.toFileName()+validation_file_name
        # generate_data(params)
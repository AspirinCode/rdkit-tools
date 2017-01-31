#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:22:56 2017

@author: Jason Shiers
"""

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from os import listdir, environ, path
from sys import exc_info
import gzip
import pickle
from time import time
import csv
from multiprocessing import Pool

class MolCollection(object):
    def __init__(self, smilesList = []):
        """
        smilesList: a list of smiles strings to load into the object
        """
        # Make copy of smilesList if this is passed in as an argument
        self.smilesList = smilesList[:] if not smilesList == None else []
        self.molList = []
        self.propertiesList = []
        self.sourceFiles = []

    def getSmilesList(self):
        return self.smilesList
        
    def getMolList(self):
        return self.molList
    
    def clearProperties(self):
        self.propertiesList = []
    
    def getPropertiesList(self):
        return self.propertiesList
    
    def setSourceFiles(self, filepath, filePrefix='', fileExt = '.b'):
        """
        Enumerates source files in which precomputed properties are stored
        filepath is the path in which the files are stored
        filePrefix will filter out files not starting with prefix
        fileExt is the extension of the appropriate files (default .b)
            will match against, for example, .b, .b000, .b001, ..., .b999
        """
        self.sourceFiles = [path.join(filepath, f) for f in listdir(filepath) if 
                            path.isfile(path.join(filepath, f)) 
                            and (f[-len(fileExt):] == fileExt 
                                 or f[-len(fileExt)-3:len(f)-3])
                            and f[:len(filePrefix)] == filePrefix]
        return self
    
    def propertiesListGenerator(self):
        """
        Generator that will sequentially load the propertiesList of each file 
        that has previously been enumerated with self.setSourceFiles
        The propertiesList is cleared and replaced when the next item is requested
        """
        if len(self.propertiesList) != 0:
            raise ValueError("Properties already loaded - check and clear first")       
        
        t = time()
        total = 0
        for file in self.sourceFiles:
            self.clearProperties()
            self.readProperties(file)
            total += len(self.propertiesList)
            m, s = divmod(time() - t, 60)
            h, m = divmod(m, 60)
            print("\rRead %.1f million items in %02d:%02d:%02ds..." % 
                  (1e-6*total, h, m, s), end="")
            yield self.propertiesList
        self.clearProperties()
            
    
    def _parse_gzSDF(self, filename, outputPrefix, filePrefix='', fastforward=0):
        """
        Private helper function for parse_gzSDF and parse_gzSDF_list
        Calculates properties and writes data every 50000 mols, 
        but leaves remainder in object for processing
        filename, outputPrefix, filePrefix and fastforward as defined in 
        bootstrap functions
        """
        if len(self.smilesList) != 0:
            raise ValueError("Object contains smiles - convert to mols")
        
        if len(self.molList) >= 50000:
            raise ValueError("Object already contains large number of unprocessed mols.")
        
        t = time()
        print("Starting with %d molecules already loaded" % (len(self.molList)))
        print("Opening: %s" % filename)
        with gzip.open(filename) as fp:
            # Assign molecule constructor to fpMols
            fpMols = Chem.ForwardSDMolSupplier(fp)
            num = 0
            
            # Fastforward over already processed mols
            while fastforward > 0:
                if next(fpMols) is not None:
                    fastforward -= 1
                    if fastforward % 50000 == 0:
                        print("\rSkipped 50000 in %d sec, %d left to skip" % 
                              (time()-t, fastforward), end='')
                        num +=1
                        t = time()
            print('')
            
            # Iterate over molecules and load into molList
            t = time()
            mol = None
            while True:
                if mol is not None:
                    self.molList.append(mol) 
                    if len(self.molList)%50000 == 0:
                        print("Imported 50000 molecules in %d sec, calculating properties" 
                              % (time()-t))
                        self.calcProperties()
                        print("Complete in %d sec" % (time()-t))
                        t = time()
                        # Write out properties for current batch of mols
                        self.writeProperties(path.join(path.dirname(filename), 
                                             outputPrefix+path.basename(filename)
                                             +".fp.b%03d" % num))
                        num += 1
                        # reinitialise object for next batch of mols
                        self.__init__()                    
                try:
                    mol = next(fpMols)
                except StopIteration:
                    print("Imported %d molecules (end of file) in %d sec" 
                          % (len(self.molList), time()-t))
                    break
    
    def parse_gzSDF(self, filename, outputPrefix, fastforward=0):
        """
        Reads in mols from a single gzipped SD file, calculates properties,
        and writes them out to disk split into files of 50000 items
        filename is for the gzipped sd file
        outputPrefix is prepended to the output file names
        fastforward can be used to skip over items if previous process was 
            interrupted in the case of large files (set as multiple of 50000)
        Output files are suffixed with .fp.b001 .fp.b002, ..., .fp.b
        """
        self._parse_gzSDF(filename, outputPrefix, fastforward)
        
        self.calcProperties()
        # Write out properties for remainder of mols
        self.writeProperties(path.join(path.dirname(filename), 
                             outputPrefix+path.basename(filename)+".fp.b"))
        self.__init__()
        return self
    
    def parse_gzSDF_list(self, filepath, outputPrefix, fileExt = '.sdf.gz'):
        """
        Reads in mols from a folder of gzipped SD files, calculates properties,
        and writes them out to disk split into files of 50000 items
        filepath is the path containing the gzipped SD files
        fileExt is the extension of the appropriate files (default .sdf.gz)
        outputPrefix is prepended to the output file names
        Output files are suffixed with .fp.b001 .fp.b002, ..., .fp.b
        """
        
        # Enumerate file list
        files = [f for f in listdir(filepath) if 
                 path.isfile(path.join(filepath, f)) 
                 and f[-len(fileExt):] == fileExt]
        
        for f in files:
            self._parse_gzSDF(path.join(filepath, f), outputPrefix)
        self.calcProperties()
        # Write out properties for remainder of mols
        self.writeProperties(path.join(path.dirname(f), 
                             outputPrefix+path.basename(f)+".fp.b"))
        self.__init__()
        return self
    
    def calcProperties(self):
        """
        Uses the mols currently loaded in object to calculate smiles and 
        fingerprint values using distributed processing (Linux only)
        """
        self.propertiesList += parallelise(getProperties, self.molList)
        return self
    
    def calcPropertiesfromSmiles(self):
        """
        Uses the smiles currently loaded in object to calculate fingerprint 
        values using distributed processing (Linux only)
        """
        self.propertiesList += parallelise(getPropertiesfromSmiles, self.smilesList)
        self.smilesList = []
        return self


    def writeProperties(self, outputFile):
        """
        Save list of (smiles, fingerprint) to outputFile (binary)
        Note that this is done iteratively as pickling large objects is very 
        memory intensive
        """
        
        if path.exists(outputFile):
            raise ValueError('Output file %s already exists' % outputFile)
        
        with open(outputFile, "wb") as fp:
            for item in self.propertiesList:
                pickle.dump(item, fp)
            print("Written to output file")
        return self
    
    def readProperties(self, inputFile):
        """
        Append propertiesList with list in file (binary)
        """
        if not path.exists(inputFile):
            raise ValueError('Input file %s does not exist' % inputFile)
    
        with open(inputFile, "rb") as fp:
            while True:
                try:
                    self.propertiesList.append(pickle.load(fp))
                except EOFError:
                    break
        return self

    def writeSmilesText(self, outputFile):
        """
        SMILES strings are written as text to outputFile (newline separated)
        """
        writeTexttoFile(outputFile, "%s\n", self.getSmilesList())
        return self
            
    def readSmilesText(self, inputFile):
        """
        Text file containing list of SMILES strings (only) is appended to smilesList
        """
        l = len(self.smilesList)
        # Append to smilesList after unpacking list of lists
        self.smilesList += [val.split(' ')[0] for 
                            sublist in readTextfromFile(inputFile) for 
                            val in sublist]

        print("Imported %d SMILES, now %d in list" 
              % (len(self.smilesList)-l, len(self.smilesList)))
        return self
    
    def writeSmiles(self, outputFile):
        """
        Save smilesList to file (binary)
        """
        writeObjtoFile(outputFile, self.getSmilesList())
        return self        
            
    def readSmiles(self, inputFile):
        """
        Read smilesList from inputFile (binary)
        """
        l = len(self.smilesList)
        self.smilesList += readObjfromFile(inputFile)
        print("Imported %d SMILES, now %d in list" 
          % (len(self.smilesList)-l, len(self.smilesList)))
        return self
        
    def getnMostSimilar(self, searchItems, n, threshold = 0):
        """
        Return n most similar structures to smiles within propertiesList
        searchItems = list of (smiles, fingerprint) to search against
        n = number of hits to return per searchItem
        threshold = minimum similarity to include
        returns dictionary:
            key: smiles of searchItem 
            value: list of n tuples (searchItem, result, similarity)
        """
        hitList = {searchItem[0]: [(None, None, 0)] * n for 
            searchItem in searchItems}
        
        for propertiesList in self.propertiesListGenerator():
            # TODO: find efficient way of searching list items for each
            # searchItem in parallel. Tried parallelise(getSimilarity, X):
            # X= [(searchItem, hitList[searchItem[0]], propertiesList) for 
            #        searchItem in searchItems]
            # Less efficient for small number of search items
            # X= [(listItem, hitList, searchItems) for listItem in propertiesList]
            # Very slow compared with non-parallel method
            #
            # TODO: Duplicate checking - break when sim ~= item in hitList
            for listItem in propertiesList:
                for searchItem in searchItems:
                    sim = getSimilarity(listItem, searchItem)[2]
                    for i in range(n):
                        if sim == None or sim < threshold:
                            break
                        if sim > hitList[searchItem[0]][i][2]:
                            hitList[searchItem[0]].insert(i, (searchItem[0], 
                                    listItem[0], sim))
                            del hitList[searchItem[0]][-1]
                            break
        print('')
        for searchItem in searchItems:
            for hit in hitList[searchItem[0]]:
                print(hit[1:])
        return hitList
    
    def __str__(self):
        string = """MolCollection\ncontains properties for %d molecules
        \rcontains %d mol files, %d smiles and points to %d source files\n""" % (
                len(self.propertiesList), len(self.molList), 
                len(self.smilesList), len(self.sourceFiles))
        return string

def calcSimilarityList(smilesList1, smilesList2, outputFile):
    """
    Calculates Tanimoto Similarity of a list of smiles pairs
    smilesPairs = tuple of (smiles1, smiles2)
    writes results to outputFile as comma separated list of:
        smiles1, smiles2, similarity
    """
    if path.exists(outputFile):
        raise ValueError('Output file %s already exists' % outputFile)
    
    results = []
    for x in range(len(smilesList1)):
        results.append((smilesList1[x], smilesList2[x], 
                        1.0 * calcTanimotoSimilarity(smilesList1[x], 
                                                     smilesList2[x])))
    # TODO: refactor using writeTexttoFile function
    with open(outputFile, "w") as fpO:
        for x in results:
            fpO.write("%s,%s,%f\n" % x)
    print("Written to output file")

# Helper Functions

def getFingerprint(mol):
    """
    Calculates rdkit Tanimoto fingerprint for mol
    """
    try:
        a = FingerprintMols.FingerprintMol(mol)
    except:
        print("Error %s - %s" % exc_info()[0:2])
        # Ensure return value can be pickled for pool
        return None
    return a

def getSmiles(mol):
    """
    Calculates rdkit smiles for mol
    """
    try:
        a = Chem.MolToSmiles(mol)
    except:
        print("Error %s - %s" % exc_info()[0:2])
        # Ensure return value can be pickled for pool
        return None
    return a

def getSimilarity(item1, item2):
    """
    Calculates similarity given two tuples of (smiles, fingerprint)
    Returns a tuple of (smiles1, smiles2, similarity)
    """
    return (item1[0], item2[0], 
            DataStructs.FingerprintSimilarity(item1[1], item2[1]))

def getProperties(mol):
    """
    Generates (smiles, fingerprint) tuple using helper functions
    """
    return (getSmiles(mol), getFingerprint(mol))

def getPropertiesfromSmiles(smiles):
    """
    Calculates rdkit Tanimoto fingerprint from smiles and returns as tuple
    """
    return (smiles, getFingerprint(Chem.MolFromSmiles(smiles)))

def parallelise(f, sequence):
    """
    Distributes a map function f on sequence over 4 child processes
    """
    # No support for forking child processes in Windows
    if environ.get('OS','') == 'Windows_NT':
        return map(f, sequence)
    
    pool = Pool(processes=4)
    result = pool.map(f, sequence)
    pool.close()
    pool.join()
    return result

def calcTanimotoSimilarity(smiles1, smiles2):
    """
    Calculates Tanimoto Similarity of two smiles strings
    """
    fps = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x)) for x in 
           [smiles1, smiles2]]
    return DataStructs.FingerprintSimilarity(fps[0],fps[1])

def writeTexttoFile(outputFile, formatString, listofValues):
    """
    Writes a list of values or list of tuples to a file, formatted by formatString
    """
    if path.exists(outputFile):
        raise ValueError('Output file %s already exists' % outputFile)
    with open(outputFile, "w") as fpO:
        for x in listofValues:
            fpO.write(formatString % x )
    print("Written to output file")

def readTextfromFile(inputFile, delim = ','):
    """
    Reads a list of values or list of tuples from a tab delimited file
    """
    if not path.exists(inputFile):
        raise ValueError('Input file %s does not exist' % inputFile)
    
    with open(inputFile, "r") as fpI:
        rows = [line for line in csv.reader(fpI, delimiter = delim)]
    return rows

def writeObjtoFile(outputFile, item):
    """
    Writes a data structure to a file using pickle
    """
    if path.exists(outputFile):
        raise ValueError('Output file %s already exists' % outputFile)

    with open(outputFile, "wb") as fpO:
        pickle.dump(item, fpO)
    print("Written to output file")

def readObjfromFile(inputFile):
    """
    Reads a data structure to a file using pickle
    """
    if not path.exists(inputFile):
        raise ValueError('Input file %s does not exist' % inputFile)

    with open(inputFile, "rb") as fpI:
        item = pickle.load(fpI)
    return item
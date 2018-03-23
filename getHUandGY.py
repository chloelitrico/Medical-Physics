import os
import numpy as np
from skimage.draw import polygon
import dicom

    
def get_ROI_Contour_Points(ROI_Name, RS_filepath):
    """
    Function: extracts the contour points of a specific ROI from an RS file
    
    Input:
    ROI_Name: name of the region of interest found in the specified RS file 
    RS_filepath: path from current directory to region where RS.dcm file is located
    
    Output:
    A 3D array of the contour points of the ROI
    """
    
    rs = dicom.read_file(RS_filepath)
    
    #Find the index at which the ROI is located in the RS file
    for index, item in enumerate(rs.StructureSetROISequence):
        if item.ROIName == ROI_Name:
            ROI_index = index
    
    #Get all contour points from RS file (organized as: [[x0-N, y0-N, z0-N][x0-N, y0-N, z0-N]] )
    contour = []
    for item in rs.ROIContourSequence[ROI_index].ContourSequence:
        contour.append(item.ContourData)
    return np.array(contour)


def convert_mm_to_index(x_vals, y_vals, z_val, image_position, pixel_spacing, slice_thickness):
    """
    Function: converts mm measurements of ROI contour points to pixel index values
    
    Input:
    x_vals: x values of ROI contour points of one slice
    y_vals: y values of ROI contour points of one slice
    z_val: z value of ROI contour points of one slice
    ***NOTE: x, y, and z values are to be in order with their point
    image_position: top left corner of the CT or RTDOSE image
    pixel_spacing: mm size of one pixel in CT or RTDOSE image
    slice_thickness: thickness of one pixel in CT or RTDOSE image
    
    Output:
    A list of x, y, and z pixel index values 
    """
    
    #Get pixel index values
    x_val_pixel_index = [abs(int(round((image_position[0] - x)/pixel_spacing[0], 0))) for x in x_vals]
    y_val_pixel_index = [abs(int(round((image_position[1] - y)/pixel_spacing[1], 0))) for y in y_vals]
    z_val_pixel_index = abs(int(round((image_position[2] - z_val)/slice_thickness, 0)))
    return x_val_pixel_index, y_val_pixel_index, z_val_pixel_index
    
def mask_per_slice(slice_contour, image_position, image_shape, pixel_spacing, slice_thickness):
    """
    Function: Makes a mask -- value: 1 inside the contour point and 0 outside contour point for each slice

    Input: 
    slice_contour: contour points of one slice
    image_position: top left corner of the CT image
    image_shape: number of pixel rows and columns of the CT image (? X ?)
    pixel_spacing: mm size of one pixel in CT image
    """
    
    #Extract X contour values of one slice 
    x_vals = slice_contour[0::3]
    #Extract Y contour values of one slice 
    y_vals = slice_contour[1::3]
    #Extract Z value of one slice 
    z_val = slice_contour[2] 
    
    x, y, z = convert_mm_to_index(x_vals, y_vals, z_val, image_position, pixel_spacing, slice_thickness)
    
    #Make an empty 2D array of 0s -- with the same size as our CT or RTDOSE image (same amount of pixel columns and rows)
    matrix = np.zeros(image_shape, dtype=np.uint8)
    #Row coordinates of vertices of polygon
    r = np.array(y)
    #Column coordinates of vertices of polygon
    c = np.array(x)
    rr, cc = polygon(r, c)
    #fill area of matrix found within volume with 1s
    matrix[rr, cc] = 1
    
    return z, matrix
 


def get_mask(ROI_Name, RS_filepath, path):
    """
    Function: makes a mask with 1: inside volume, 0: outside volume depending on contour points of an ROI
    
    Input:
    ROI_Name: name of the region of interest found within the RS file
    RS_filepath: filepath to the RS folder
    path: path to RTDOSE file or CTSCAN folder
    
    Output: 
    mask: matrix containing 1s (region inside ROI contour) and 0s (outside ROI contour)
    slice_numbers: z-value of slices in order of the 3D mask 
    """

    #if CT folderpath is given
    if path[-4:] != '.dcm':

        #Obtain a list of the files found within the CT folder
        CT_slices = os.listdir(path)
    
        #Assuming all CT images in the same folder are of the same shape and position
        file = dicom.read_file(path + '/' + CT_slices[0])
    
    #if RTDOSE filepath is given
    else:
        file = dicom.read_file(path)
    
    slice_numbers = []
    mask = []
    #Get number of pixel rows and columns of the RD image
    image_shape = (file.Rows, file.Columns)
    #Get top left corner of CT image
    image_position = [float(x) for x in file.ImagePositionPatient]
    pixel_spacing = [float(x) for x in file.PixelSpacing]
    slice_thickness = float(file.SliceThickness)

    #Get matrix for each slice and create 3D matrix
    for slice_contour in get_ROI_Contour_Points(ROI_Name, RS_filepath):
        slice_n, m = mask_per_slice(slice_contour, image_position, image_shape, pixel_spacing, slice_thickness)
        if slice_n not in slice_numbers:
            slice_numbers.append(slice_n)
            mask.append(m)
    
    #To check if slices are in order
    if slice_numbers != sorted(slice_numbers):
        slice_numbers, mask = [list(t) for t in (zip(*sorted(zip(slice_numbers, mask))))]        

    return np.array(mask), slice_numbers



def get_gy(RD_filepath, mask, slice_numbers):

    #Open RD files that are found wihtin the mask -- convert pixel values to hounsfield units
    ROI_gy_vals = []

    rd = dicom.read_file(RD_filepath)
    pixel_image = rd.pixel_array
    gy_image = pixel_image * rd.DoseGridScaling
        
    #go through each value in the slice, and save values which are found within the ROI
    for z, contour_slice in enumerate(mask):
        for i, row in enumerate(contour_slice):
            for j, voxel in enumerate(row):
                if voxel == 1:
                    ROI_gy_vals.append(gy_image[slice_numbers[z]][i][j])
    return ROI_gy_vals


def get_hu(CT_folderpath, mask, slice_numbers):
    """
    Function: returns a list of the hounsfield units found within the mask
    
    Input:
    CT_folderpath: path to CT folder containing all CT images of each slice
    mask: 3D matrix contianing 1s (volume inside ROI), and 0 (volume outside ROI)
    slice_numbers: the z value of slice found within the ROI (the z values of each slice of the mask)
    
    Output: 
    ROI_hu_vals: list of all hounsfield units found within the ROI
    """
    
    #list of names of all CT files in the folder containing CT files of one CT scan
    CT_files = os.listdir(CT_folderpath)

    slice_location = []
    CT_slices = []

    #only save CT_file slices that are found within the mask
    for CT_file in CT_files:
        ct = dicom.read_file(CT_folderpath + '/' + CT_file) 

        slice_loc = ct.InstanceNumber

        if slice_loc in slice_numbers:
            slice_location.append(slice_loc)
            CT_slices.append(CT_file)

    #To check if slices are in order -- puts them in order if they are not
    if slice_location != sorted(slice_location):
        slice_location, CT_slices = (list(t) for t in zip(*sorted(zip(slice_location, CT_slices))))
    
    #Open CT files that are found wihtin the mask -- convert pixel values to hounsfield units
    ROI_hu_vals = []

    for i, contour_slice in enumerate(mask):
        ds = dicom.read_file(CT_folderpath + '/' + CT_slices[i])
        pixel_image = ds.pixel_array
        slope = ds.RescaleSlope
        intercept = ds.RescaleIntercept
        hu_image = pixel_image * slope + intercept
        
        #go through each value in the slice, and save values which are found within the ROI
        for i, row in enumerate(contour_slice):
            for j, voxel in enumerate(row):
                if voxel == 1:
                    ROI_hu_vals.append(hu_image[i][j])
    return ROI_hu_vals


def get_HU_and_GY_txt(RS_filepath, CT_folderpath, RD_filepath, ROI_Name, output_folderpath, patient_x):
    """
    Function: returns a list of the hounsfield units in a txt file
    
    Input:
    RS_filepath: filepath to the RTSTRUCT file
    ROI_Name: name of the region of interest found within the RS file
    RD_filepath: filepath to the RTDOSE file
    CT_folderpath: path to CT folder containing all CT images of each slice
    output_folderpath: path to specified folder/creates folder that will contain all HU.txt files
    patient_x: patient number -- to identify from other patients 
    
    Output: 
    A txt file with HU vals
    """

    #Get GY mask and slice_numbers
    gy_mask, gy_slice_numbers = get_mask(ROI_Name, RS_filepath, RD_filepath)
    #Get GY
    gy = get_gy(RD_filepath, gy_mask, gy_slice_numbers)

    #Get HU mask and slice_numbers
    hu_mask, hu_slice_numbers = get_mask(ROI_Name, RS_filepath, CT_folderpath)
    #Get HU
    hounsfield_units = get_hu(CT_folderpath, hu_mask, hu_slice_numbers)
    
    
    #Assuming all CT images in the same folder are of the same shape and position    
    ct = dicom.read_file(CT_folderpath + '/' + os.listdir(CT_folderpath)[0])
    rd = dicom.read_file(RD_filepath)
    
    #check if output folder exists
    directory = os.path.dirname(output_folderpath + "/" + patient_x + "_" + rd.StudyDate + ".txt")
    #If the output folder does not exist, create folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    #write HUs to txt file
    f = open(output_folderpath + "/yoooo" + patient_x + "_" + rd.StudyDate + ".txt",'w')
    f.write("Hounsfield Units: \n" + ', '.join(str(x) for x in hounsfield_units)+ "\n")
    f.write("GY: \n" + ', '.join(str(x) for x in gy)+ "\n")
    
    #Save certain values to txt file
    f.write("Patient's Birth Year: \n" + rd.PatientsBirthDate[0:3] + "\n")
    f.write("Voxel Diameter for CTSCAN (x, y, z): \n"  + ", ".join(str(x) for x in ct.PixelSpacing) + ", " + str(ct.SliceThickness) + "\n")
    f.write("Voxel Diameter for RTDOSE (x, y, z): \n"  + ", ".join(str(x) for x in rd.PixelSpacing) + ", " + str(rd.SliceThickness) + "\n")
    f.close()
import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.write("PIL not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image as PILImage
    PIL_AVAILABLE = True

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.write("qrcode not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qrcode'])
    import qrcode
    QR_AVAILABLE = True

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

def format_number_smartly(value):
    """
    Smart number formatting: integers display as integers, decimals display as decimals
    """
    if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'none', 'null']:
        return ''
    
    try:
        # Convert to float to handle numeric operations
        num_value = float(value)
        
        # Check if it's actually an integer (no decimal part)
        if num_value.is_integer():
            return str(int(num_value))
        else:
            # Format decimal to remove unnecessary trailing zeros
            formatted = f"{num_value:g}"  # 'g' removes trailing zeros
            return formatted
    except (ValueError, TypeError):
        # If it's not a number, return as string
        return str(value).strip()

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    cols = [str(col).upper() for col in df_columns]
    
    # Priority order for bus model column detection
    patterns = [
        # Exact matches (highest priority)
        lambda col: col == 'BUS_MODEL',
        lambda col: col == 'BUSMODEL',
        lambda col: col == 'BUS MODEL',
        lambda col: col == 'MODEL',
        lambda col: col == 'BUS_TYPE',
        lambda col: col == 'BUSTYPE',
        lambda col: col == 'BUS TYPE',
        lambda col: col == 'VEHICLE_TYPE',
        lambda col: col == 'VEHICLETYPE',
        lambda col: col == 'VEHICLE TYPE',
        # Partial matches (lower priority)
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            if pattern(col):
                return df_columns[i]  # Return original column name
    
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    # Get quantity value with smart formatting
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = format_number_smartly(row[qty_veh_col])
    
    if not qty_veh:
        return result
    
    # Method 1: Check if quantity already contains model info (e.g., "9M:2", "7M-3", "12M 5")
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field itself
        for model, quantity in matches:
            if model in result:
                result[model] = format_number_smartly(quantity)
        return result
    
    # Method 2: Look for bus model in dedicated bus model column first
    detected_model = None
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        
        # Check for exact matches first
        if bus_model_value in ['7M', '7']:
            detected_model = '7M'
        elif bus_model_value in ['9M', '9']:
            detected_model = '9M'
        elif bus_model_value in ['12M', '12']:
            detected_model = '12M'
        # Check for patterns within the text
        elif re.search(r'\b7M\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9M\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12M\b', bus_model_value):
            detected_model = '12M'
        # Check for standalone numbers
        elif re.search(r'\b7\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12\b', bus_model_value):
            detected_model = '12M'
    
    # If we found a model in the dedicated column, use it
    if detected_model:
        result[detected_model] = qty_veh
        return result
    
    # Method 3: Search through all columns systematically with priority
    # First, search in columns that are most likely to contain bus model info
    priority_columns = []
    other_columns = []
    
    for col in row.index:
        if pd.notna(row[col]):
            col_upper = str(col).upper()
            # High priority columns
            if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                priority_columns.append(col)
            else:
                other_columns.append(col)
    
    # Search priority columns first
    for col in priority_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Look for exact matches first
            if re.search(r'\b7M\b', value_str):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9M\b', value_str):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12M\b', value_str):
                result['12M'] = qty_veh
                return result
            # Then look for standalone numbers in context
            elif re.search(r'\b7\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['12M'] = qty_veh
                return result
    
    # Method 4: Search in other columns as fallback
    detected_models = []
    for col in other_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Use word boundaries to avoid false matches
            if re.search(r'\b7M\b', value_str):
                detected_models.append('7M')
            elif re.search(r'\b9M\b', value_str):
                detected_models.append('9M')
            elif re.search(r'\b12M\b', value_str):
                detected_models.append('12M')
    
    # Remove duplicates while preserving order
    detected_models = list(dict.fromkeys(detected_models))
    
    if detected_models:
        # Use the first detected model
        result[detected_models[0]] = qty_veh
        return result
    
    # Method 5: Last resort - look for standalone numbers that might indicate bus length
    for col in row.index:
        if pd.notna(row[col]):
            value_str = format_number_smartly(row[col])
            
            # Look for exact matches of just the number
            if value_str == '7':
                result['7M'] = qty_veh
                return result
            elif value_str == '9':
                result['9M'] = qty_veh
                return result
            elif value_str == '12':
                result['12M'] = qty_veh
                return result
    
    # Method 6: If still no model detected, return empty (no boxes filled)
    return result

def generate_qr_code(data_string):
    """
    Generate a QR code from the given data string
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        
        # Add data
        qr.add_data(data_string)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes that reportlab can use
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create a QR code image with specified size
        return Image(img_buffer, width=2.5*cm, height=2.5*cm)
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_location_string(location_str):
    """Parse a location string into components for table display"""
    # Initialize with empty values
    location_parts = [''] * 7

    if not location_str or not isinstance(location_str, str):
        return location_parts

    # Remove any extra spaces and handle NaN values
    location_str = str(location_str).strip()
    if location_str.lower() in ['nan', 'none', 'null']:
        return location_parts

    # Try to parse location components
    import re
    pattern = r'([^_\s]+)'
    matches = re.findall(pattern, location_str)

    # Fill the available parts
    for i, match in enumerate(matches[:7]):
        if match.lower() not in ['nan', 'none', 'null']:
            location_parts[i] = match

    return location_parts

def extract_location_data_from_excel(row_data):
    """Extract location data from Excel row for Line Location"""
    # Get all available columns for debugging
    available_cols = list(row_data.index) if hasattr(row_data, 'index') else []
    
    # Try different variations of column names (case-insensitive)
    def find_column_value(possible_names, default=''):
        for name in possible_names:
            # Try exact match first
            if name in row_data:
                val = row_data[name]
                return format_number_smartly(val) if pd.notna(val) and str(val).lower() != 'nan' else default
            # Try case-insensitive match
            for col in available_cols:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    return format_number_smartly(val) if pd.notna(val) and str(val).lower() != 'nan' else default
        return default
    
    # Extract values with multiple possible column names
    bus_model = find_column_value(['Bus Model', 'Bus model', 'BUS MODEL', 'BUSMODEL', 'Bus_Model'])
    station_no = find_column_value(['Station No', 'Station no', 'STATION NO', 'STATIONNO', 'Station_No'])
    rack = find_column_value(['Rack', 'RACK', 'rack'])
    rack_no_1st = find_column_value(['Rack No (1st digit)', 'RACK NO (1st digit)', 'Rack_No_1st', 'RACK_NO_1ST'])
    rack_no_2nd = find_column_value(['Rack No (2nd digit)', 'RACK NO (2nd digit)', 'Rack_No_2nd', 'RACK_NO_2ND'])
    level = find_column_value(['Level', 'LEVEL', 'level'])
    cell = find_column_value(['Cell', 'CELL', 'cell'])
    
    return [bus_model, station_no, rack, rack_no_1st, rack_no_2nd, level, cell]


def extract_store_location_data_from_excel(row_data):
    """Extract store location data from Excel row for Store Location"""
    def get_clean_value(possible_names, default=''):
        """Get clean value from multiple possible column names"""
        for name in possible_names:
            # Try exact match first
            if name in row_data:
                val = row_data[name]
                if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                    return format_number_smartly(val)
            # Try case-insensitive match
            for col in row_data.index:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                        return format_number_smartly(val)
        return default
    
    # Extract values with proper column name handling
    # First cell: Station Name
    station_name = get_clean_value(['Station Name', 'STATION NAME', 'Station_Name', 'STATIONNAME'], '')
    
    # Second cell: Store Location
    store_location = get_clean_value(['Store Location', 'STORE LOCATION', 'Store_Location', 'STORELOCATION'], '')
    
    # Remaining cells: ABB values
    zone = get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE'], '')
    location = get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'], '')
    floor = get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'], '')
    rack_no = get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO'], '')
    level_in_rack = get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'], '')
    
    return [station_name, store_location, zone, location, floor, rack_no, level_in_rack]

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data"""
    if status_callback:
        status_callback(f"Processing file: {excel_file_path}")
    else:
        st.write(f"Processing file: {excel_file_path}")

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area (10cm x 7.5cm)
        # Position it at the top of the page with minimal margin
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm  # Position at top with minimal margin
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.95))  # Slightly darker black (95% opacity)
        canvas.setLineWidth(1.8)  # Slightly thicker border
        canvas.rect(
            x_offset + doc.leftMargin,
            y_offset,
            CONTENT_BOX_WIDTH - 0.2*cm,  # Account for margins
            CONTENT_BOX_HEIGHT
        )
        canvas.restoreState()

    # Load the Excel data
    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            try:
                df = pd.read_excel(excel_file_path)
            except Exception as e:
                try:
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                except Exception as e2:
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_callback:
            status_callback(f"Successfully read file with {len(df)} rows")
            status_callback(f"Columns found: {df.columns.tolist()}")
        else:
            st.write(f"Successfully read file with {len(df)} rows")
            st.write("Columns found:", df.columns.tolist())
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        return None

    # Identify columns (case-insensitive)
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = next((col for col in cols if 'QTY/BIN' in col or 'QTY_BIN' in col or 'QTYBIN' in col), 
                  next((col for col in cols if 'QTY' in col and 'BIN' in col), None))
    
    # If no specific QTY/BIN column is found, fall back to general QTY column
    if not qty_bin_col:
        qty_bin_col = next((col for col in cols if 'QTY' in col),
                      next((col for col in cols if 'QUANTITY' in col), None))
  
    loc_col = next((col for col in cols if 'LOC' in col or 'POS' in col or 'LOCATION' in col),
                   cols[2] if len(cols) > 2 else desc_col)

    # Improved detection of QTY/VEH column
    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    # Look for store location column
    store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                      next((col for col in cols if 'STORELOCATION' in col), None))

    # Find bus model column using the enhanced detection function
    bus_model_col = find_bus_model_column(original_columns)

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            status_callback(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            st.write(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")

    # Create document with minimal margins
    doc = SimpleDocTemplate(output_pdf_path, pagesize=STICKER_PAGESIZE,
                          topMargin=0.2*cm,  # Minimal top margin
                          bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
                          leftMargin=0.1*cm, rightMargin=0.1*cm)

    content_width = CONTENT_BOX_WIDTH - 0.2*cm
    all_elements = []
    
    # ✅ INSERT SORTING HERE
    rack_col = next((col for col in df.columns if col.strip().lower() == 'rack'), None)
    rack_no_1st_col = next((col for col in df.columns if '1st' in col.lower()), None)
    rack_no_2nd_col = next((col for col in df.columns if '2nd' in col.lower()), None)

    if rack_col and rack_no_1st_col and rack_no_2nd_col:
        df[rack_no_1st_col] = pd.to_numeric(df[rack_no_1st_col], errors='coerce')
        df[rack_no_2nd_col] = pd.to_numeric(df[rack_no_2nd_col], errors='coerce')

        df.sort_values(
            by=[rack_col, rack_no_1st_col, rack_no_2nd_col],
            ascending=[False, False, False],
            inplace=True
        )
    else:
        if status_callback:
            status_callback("⚠️ Sorting skipped: could not find all rack-related columns.")
        else:
            st.warning("⚠️ Sorting skipped: could not find all rack-related columns.")


    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if status_callback:
            status_callback(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data with smart formatting
        part_no = format_number_smartly(row[part_no_col])
        desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
        
        # Extract QTY/BIN properly with smart formatting
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = format_number_smartly(row[qty_bin_col])
            
        # Extract QTY/VEH properly with smart formatting
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = format_number_smartly(row[qty_veh_col])
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row and pd.notna(row[loc_col]) else ""
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row and pd.notna(row[store_loc_col]) else ""
        location_parts = parse_location_string(location_str)

        # Use enhanced bus model detection
        mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
        qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}"
        
        qr_image = generate_qr_code(qr_data)
        if status_callback and qr_image:
            status_callback(f"QR code generated for part: {part_no}")
        
        # Define row heights
        header_row_height = 0.9*cm
        desc_row_height = 1.0*cm
        qty_row_height = 0.5*cm
        location_row_height = 0.5*cm

        # Main table data
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, desc_style)],
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, qty_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(main_table)

       # Store Location section
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
        name='StoreLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        # Total width for the 7 inner columns (2/3 of full content width)
        inner_table_width = content_width * 2 / 3

        # Define proportional widths - same as Line Location for consistency
        col_proportions = [1.5, 2.5, 0.7, 0.8, 0.8, 0.7, 0.9]
        total_proportion = sum(col_proportions)

        # Calculate column widths based on proportions 
        inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

        # Extract store location values from Excel data
        store_loc_values = extract_store_location_data_from_excel(row)

        store_loc_inner_table = Table(
            [store_loc_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        store_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make store location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )
        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(store_loc_table)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        
        # Extract line location values from Excel data
        line_loc_values = extract_location_data_from_excel(row)
        
        line_loc_inner_table = Table(
            [line_loc_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        line_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        line_loc_table = Table(
            [[line_loc_label, line_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )
        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(line_loc_table)

        # MTM section with QR code
        mtm_label = Paragraph("MTM", ParagraphStyle(
            name='MTM', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        # Create MTM boxes table
        mtm_boxes_data = [
            ["7M", "9M", "12M"],
            [mtm_quantities['7M'], mtm_quantities['9M'], mtm_quantities['12M']]
        ]

        mtm_boxes_table = Table(mtm_boxes_data,
                               colWidths=[inner_table_width/3] * 3,
                               rowHeights=[0.4*cm, 0.6*cm])

        mtm_boxes_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 12),
        ]))

        # Create bottom section with MTM and QR code
        if qr_image:
            bottom_table_data = [[mtm_label, mtm_boxes_table, qr_image]]
            bottom_col_widths = [content_width/6, inner_table_width*2/3, content_width/6]
        else:
            bottom_table_data = [[mtm_label, mtm_boxes_table]]
            bottom_col_widths = [content_width/3, inner_table_width]

        bottom_table = Table(bottom_table_data,
                           colWidths=bottom_col_widths,
                           rowHeights=[1.0*cm])

        bottom_table.setStyle(TableStyle([
            ('GRID', (0, 0), (1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(bottom_table)

        # Add all elements to main content list
        all_elements.extend(elements)
        
        # Add page break except for the last item
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build PDF with border
    try:
        doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
        if status_callback:
            status_callback(f"✅ PDF generated successfully: {output_pdf_path}")
        else:
            st.success(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error generating PDF: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main Streamlit app function"""
    st.title("📋 Sticker Label Generator")
    st.write("Upload an Excel file to generate sticker labels with QR codes")

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Output file path
        output_path = "sticker_labels.pdf"

        # Generate labels button
        if st.button("🏷️ Generate Sticker Labels"):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_status(message):
                status_text.text(message)
                if "%" in message:
                    try:
                        percent = int(message.split("(")[1].split("%")[0])
                        progress_bar.progress(percent / 100)
                    except:
                        pass

            # Generate the PDF
            result_path = generate_sticker_labels(tmp_file_path, output_path, update_status)
            
            if result_path:
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                # Provide download link
                with open(result_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_file.read(),
                        file_name="sticker_labels.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Failed to generate PDF. Please check the error messages above.")

        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

    # Instructions
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. **Upload your Excel file** containing part information
    2. **Required columns** (case-insensitive):
       - Part Number column (containing 'PART' and 'NO'/'NUM'/'#')
       - Description column (containing 'DESC' or 'NAME')
       - Location column (containing 'LOC'/'POS'/'LOCATION')
       - Qty/Bin column (containing 'QTY')
    3. **Optional columns**:
       - Qty/Veh column (for vehicle-specific quantities)
       - Store Location column (for store positioning)
       - Bus Model column (for MTM box filling)
    4. **Click Generate** to create your sticker labels
    5. **Download** the generated PDF file
    
    The app will automatically detect column names and generate QR codes for each part.
    """)

if __name__ == "__main__":
    main()

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

def clean_numeric_value(value):
    """Convert decimal numbers to integers if they are whole numbers, otherwise keep as is"""
    if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'none', 'null']:
        return ''
    
    try:
        # Try to convert to float first
        float_val = float(value)
        # If it's a whole number, convert to int
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return str(float_val)
    except (ValueError, TypeError):
        # If conversion fails, return as string
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

def find_column_flexible(df_columns, search_terms):
    """
    Flexible column finder that handles spaces, parentheses, and case variations
    """
    # Normalize column names for searching
    normalized_cols = []
    for col in df_columns:
        # Remove spaces, parentheses, and convert to uppercase
        normalized = re.sub(r'[^\w]', '', str(col).upper())
        normalized_cols.append(normalized)
    
    # Normalize search terms
    normalized_terms = []
    for term in search_terms:
        normalized = re.sub(r'[^\w]', '', term.upper())
        normalized_terms.append(normalized)
    
    # Find matches
    for term in normalized_terms:
        for i, normalized_col in enumerate(normalized_cols):
            if term == normalized_col:
                return df_columns[i]
    
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col]).strip()
    
    if not qty_veh:
        return result
    
    # Method 1: Check if quantity already contains model info (e.g., "9M:2", "7M-3", "12M 5")
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field itself
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
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
            value_str = str(row[col]).strip()
            
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
    location_str = location_str.strip()
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
    """Extract location data from Excel row for Line Location with improved column detection"""
    # Get all available columns for debugging
    available_cols = list(row_data.index) if hasattr(row_data, 'index') else []
    
    # Try different variations of column names (case-insensitive and flexible)
    def find_column_value(possible_names, default=''):
        for name in possible_names:
            # Try exact match first
            if name in row_data:
                val = row_data[name]
                return clean_numeric_value(val) if pd.notna(val) else default
            # Try case-insensitive match
            for col in available_cols:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    return clean_numeric_value(val) if pd.notna(val) else default
            # Try flexible matching (removing spaces and special characters)
            normalized_name = re.sub(r'[^\w]', '', name.upper())
            for col in available_cols:
                normalized_col = re.sub(r'[^\w]', '', str(col).upper())
                if normalized_col == normalized_name:
                    val = row_data[col]
                    return clean_numeric_value(val) if pd.notna(val) else default
        return default
    
    # Extract values with multiple possible column names including flexible matching
    bus_model = find_column_value(['Bus Model', 'Bus model', 'BUS MODEL', 'BUSMODEL', 'Bus_Model'])
    station_no = find_column_value(['Station No', 'Station no', 'STATION NO', 'STATIONNO', 'Station_No'])
    rack = find_column_value(['Rack', 'RACK', 'rack'])
    
    # Enhanced detection for rack numbers with spaces and parentheses
    rack_no_1st = find_column_value([
        'Rack No (1st digit)', 'RACK NO (1st digit)', 'Rack_No_1st', 'RACK_NO_1ST',
        'RACK NO ( 1 st digit )', 'Rack No ( 1 st digit )',
        'RACKNO1STDIGIT', 'RACK NO 1ST DIGIT', 'RACKNO1ST'
    ])
    
    rack_no_2nd = find_column_value([
        'Rack No (2nd digit)', 'RACK NO (2nd digit)', 'Rack_No_2nd', 'RACK_NO_2ND',
        'RACK NO ( 2 nd digit )', 'Rack No ( 2 nd digit )',
        'RACKNO2NDDIGIT', 'RACK NO 2ND DIGIT', 'RACKNO2ND'
    ])
    
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
                    return clean_numeric_value(val)
            # Try case-insensitive match
            for col in row_data.index:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                        return clean_numeric_value(val)
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

def create_sorting_key(row, rack_1st_col, rack_2nd_col):
    """Create a sorting key for rack numbers to ensure proper ordering"""
    rack_1st = ''
    rack_2nd = ''
    
    if rack_1st_col and rack_1st_col in row:
        rack_1st = clean_numeric_value(row[rack_1st_col])
    
    if rack_2nd_col and rack_2nd_col in row:
        rack_2nd = clean_numeric_value(row[rack_2nd_col])
    
    # Convert to integers for proper sorting, default to 0 if empty or non-numeric
    try:
        rack_1st_int = int(rack_1st) if rack_1st else 0
    except ValueError:
        rack_1st_int = 0
    
    try:
        rack_2nd_int = int(rack_2nd) if rack_2nd else 0
    except ValueError:
        rack_2nd_int = 0
    
    return (rack_1st_int, rack_2nd_int)

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

    # Find rack columns for sorting using flexible detection
    rack_1st_col = find_column_flexible(original_columns, [
        'RACK NO ( 1 st digit )', 'RACK NO (1st digit)', 'RACK_NO_1ST', 'RACKNO1STDIGIT'
    ])
    
    rack_2nd_col = find_column_flexible(original_columns, [
        'RACK NO ( 2 nd digit )', 'RACK NO (2nd digit)', 'RACK_NO_2ND', 'RACKNO2NDDIGIT'
    ])

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            status_callback(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
        if rack_1st_col:
            status_callback(f"Rack 1st Digit Column: {rack_1st_col}")
        if rack_2nd_col:
            status_callback(f"Rack 2nd Digit Column: {rack_2nd_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            st.write(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")
        if rack_1st_col:
            st.write(f"Rack 1st Digit Column: {rack_1st_col}")
        if rack_2nd_col:
            st.write(f"Rack 2nd Digit Column: {rack_2nd_col}")

    # Sort the dataframe by rack numbers if rack columns are found
    if rack_1st_col or rack_2nd_col:
        if status_callback:
            status_callback("Sorting data by rack numbers...")
        
        # Create sorting keys
        df['_sort_key'] = df.apply(lambda row: create_sorting_key(row, rack_1st_col, rack_2nd_col), axis=1)
        
        # Sort by the sorting key
        df = df.sort_values('_sort_key')
        
        # Remove the temporary sorting column
        df = df.drop(columns=['_sort_key'])
        
        if status_callback:
            status_callback("Data sorted successfully by rack numbers")

    # Create document with minimal margins
    doc = SimpleDocTemplate(output_pdf_path, pagesize=STICKER_PAGESIZE,
                          topMargin=0.2*cm,  # Minimal top margin
                          bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
                          leftMargin=0.1*cm, rightMargin=0.1*cm)

    content_width = CONTENT_BOX_WIDTH - 0.2*cm
    all_elements = []

    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if status_callback:
            status_callback(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = clean_numeric_value(row[qty_bin_col])
            
        # Extract QTY/VEH properly
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = clean_numeric_value(row[qty_veh_col])
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row else ""
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row else ""
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
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 10),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica'),
            ('FONTSIZE', (1, 1), (1, 1), 10),
            ('FONTNAME', (1, 2), (1, 2), 'Helvetica'),
            ('FONTSIZE', (1, 2), (1, 2), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        elements.append(main_table)
        elements.append(Spacer(1, 0.1*cm))

        # Create location table based on whether we have store location data
        if store_location:
            # Store Location format
            location_data = extract_store_location_data_from_excel(row)
            location_table_data = [
                ["Store Location"] + location_data
            ]
        else:
            # Line Location format  
            location_data = extract_location_data_from_excel(row)
            location_table_data = [
                ["Line Location", "Station No", "Rack", "Rack No (1st digit)", "Rack No (2nd digit)", "Level", "Cell"],
                location_data
            ]

        # Calculate column widths for location table
        if store_location:
            num_cols = len(location_table_data[0])
            location_col_widths = [content_width / num_cols] * num_cols
        else:
            location_col_widths = [content_width/7] * 7

        location_table = Table(location_table_data,
                             colWidths=location_col_widths,
                             rowHeights=[location_row_height] * len(location_table_data))

        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        elements.append(location_table)
        elements.append(Spacer(1, 0.1*cm))

        # MTM table with QR code
        mtm_table_data = [
            ["MTM", "7M", "9M", "12M", "QR Code"],
            ["", mtm_quantities['7M'], mtm_quantities['9M'], mtm_quantities['12M'], ""]
        ]

        # Calculate column widths for MTM table
        qr_col_width = 2.8*cm
        remaining_width = content_width - qr_col_width
        mtm_col_widths = [remaining_width/4] * 4 + [qr_col_width]

        mtm_table = Table(mtm_table_data, colWidths=mtm_col_widths, 
                         rowHeights=[0.5*cm, 1.5*cm])

        mtm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        elements.append(mtm_table)

        # Add QR code to the MTM table if it was generated successfully
        if qr_image:
            # Create a new table with the QR code positioned correctly
            qr_table_data = [
                ["MTM", "7M", "9M", "12M", qr_image],
                ["", mtm_quantities['7M'], mtm_quantities['9M'], mtm_quantities['12M'], ""]
            ]

            qr_table = Table(qr_table_data, colWidths=mtm_col_widths, 
                           rowHeights=[0.5*cm, 1.5*cm])

            qr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('SPAN', (4, 0), (4, 1)),  # Span QR code cell across both rows
            ]))

            # Replace the last element (MTM table without QR) with the QR version
            elements[-1] = qr_table

        # Add all elements for this sticker
        all_elements.extend(elements)
        
        # Add page break after each sticker except the last one
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build PDF with border
    if status_callback:
        status_callback("Building final PDF...")
    
    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    
    if status_callback:
        status_callback(f"PDF generated successfully: {output_pdf_path}")
    else:
        st.success(f"PDF generated successfully: {output_pdf_path}")
    
    return output_pdf_path

def main():
    st.title("Sticker Label Generator")
    st.write("Upload an Excel or CSV file to generate sticker labels with QR codes")

    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Generate output filename
        output_filename = f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Create progress container
        progress_container = st.empty()
        status_container = st.empty()
        
        def update_status(message):
            status_container.write(f"Status: {message}")
        
        try:
            # Generate stickers
            update_status("Starting sticker generation...")
            result_path = generate_sticker_labels(tmp_file_path, output_path, update_status)
            
            if result_path and os.path.exists(result_path):
                # Provide download button
                with open(result_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="Download Sticker Labels PDF",
                    data=pdf_bytes,
                    file_name=output_filename,
                    mime="application/pdf"
                )
                
                st.success("Stickers generated successfully! Click the download button above.")
                
        except Exception as e:
            st.error(f"Error generating stickers: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
        
        finally:
            # Clean up temporary files
            try:
                os.unlink(tmp_file_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass

if __name__ == "__main__":
    main()

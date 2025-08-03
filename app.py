import streamlit as st
import numpy as np
from PIL import Image
import io
import colorsys

st.title("🧠 Sparse HSL Rectangular Encoder")
st.write("Upload a PNG image and compare standard RGB size vs. optimized HSL rectangular format.")

if 'conversion_complete' not in st.session_state:
    st.session_state.conversion_complete = False
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'compression_results' not in st.session_state:
    st.session_state.compression_results = None
if 'clear_uploader' not in st.session_state:
    st.session_state.clear_uploader = False

uploaded_file = st.file_uploader("Upload PNG Image", type=["png"], key=f"uploader_{st.session_state.clear_uploader}")

if uploaded_file and not st.session_state.conversion_complete:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape
    
    st.session_state.image_data = {
        'image': image,
        'img_np': img_np,
        'h': h,
        'w': w
    }
    
    st.image(image, caption=f"Original Image ({w}x{h})", use_column_width=True)
    
    st.subheader("🎛️ Compression Settings")
    use_reduced_precision = st.checkbox(
        "Use Reduced HSL Precision", 
        value=False,
        help="Reduces HSL ranges: H: 360°→256°, S: 100°→64°, L: 100°→64° for better compression"
    )
    
    if use_reduced_precision:
        st.info("📊 **Reduced Precision Mode**: H(0-255), S(0-63), L(0-63) - Fewer unique values = Better compression")
    else:
        st.info("📊 **Full Precision Mode**: H(0-359), S(0-99), L(0-99) - Maximum quality")
    
    if st.button("🚀 Convert Image", type="primary"):
        st.session_state.reduced_precision = use_reduced_precision
        st.session_state.conversion_complete = True
        st.rerun()

elif st.session_state.image_data:
    image = st.session_state.image_data['image']
    img_np = st.session_state.image_data['img_np']
    h = st.session_state.image_data['h']
    w = st.session_state.image_data['w']
    
    st.image(image, caption=f"Original Image ({w}x{h})", use_column_width=True)
    
    if st.button("🔄 Convert Different Image"):
        st.session_state.conversion_complete = False
        st.session_state.image_data = None
        st.session_state.compression_results = None
        st.session_state.clear_uploader = not st.session_state.clear_uploader
        st.rerun()

if st.session_state.conversion_complete and st.session_state.image_data:
    if not st.session_state.compression_results:
        image = st.session_state.image_data['image']
        img_np = st.session_state.image_data['img_np']
        h = st.session_state.image_data['h']
        w = st.session_state.image_data['w']
        
        st.subheader("🎨 RGB to HSL Conversion")
        progress_conversion = st.progress(0)
        conversion_text = st.empty()
        
        hsl_img = np.zeros_like(img_np, dtype=np.float32)
        total_pixels = h * w
        
        use_reduced = st.session_state.get('reduced_precision', False)
        
        for y in range(h):
            for x in range(w):
                r, g, b = img_np[y, x] / 255.0
                hsl_h, hsl_l, hsl_s = colorsys.rgb_to_hls(r, g, b)
                
                if use_reduced:
                    # Reduced precision: H(0-255), S(0-63), L(0-63)
                    h_val = hsl_h * 255.0  # 0-255 instead of 0-359
                    s_val = hsl_s * 63.0   # 0-63 instead of 0-99
                    l_val = hsl_l * 63.0   # 0-63 instead of 0-99
                else:
                    # Full precision: H(0-359), S(0-99), L(0-99)
                    h_val = hsl_h * 360.0
                    s_val = hsl_s * 100.0
                    l_val = hsl_l * 100.0
                
                hsl_img[y, x] = [h_val, s_val, l_val]
                
                if (y * w + x) % 1000 == 0:
                    progress_conversion.progress((y * w + x) / total_pixels)
                    conversion_text.text(f"Converting pixel {y * w + x}/{total_pixels}")
        
        progress_conversion.progress(1.0)
        conversion_text.text("HSL conversion complete!")
        
        if use_reduced:
            H = np.round(hsl_img[:, :, 0]).astype(np.uint8)   # H: 0-255
            S = np.round(hsl_img[:, :, 1]).astype(np.uint8)   # S: 0-63
            L = np.round(hsl_img[:, :, 2]).astype(np.uint8)   # L: 0-63
            st.success("🎯 Using Reduced Precision: H(0-255), S(0-63), L(0-63)")
        else:
            H = np.round(hsl_img[:, :, 0]).astype(np.uint16)  # H: 0-359
            S = np.round(hsl_img[:, :, 1]).astype(np.uint8)   # S: 0-99
            L = np.round(hsl_img[:, :, 2]).astype(np.uint8)   # L: 0-99
            st.info("📏 Using Full Precision: H(0-359), S(0-99), L(0-99)")

        unique_H = np.unique(H)
        unique_S = np.unique(S)
        unique_L = np.unique(L)

        st.markdown(f"**HSL Channel Analysis:**")
        st.markdown(f"- Hue (H): {len(unique_H)} unique values (range: {unique_H.min()}-{unique_H.max()})")
        st.markdown(f"- Saturation (S): {len(unique_S)} unique values (range: {unique_S.min()}-{unique_S.max()})")
        st.markdown(f"- Lightness (L): {len(unique_L)} unique values (range: {unique_L.min()}-{unique_L.max()})")
        
        if use_reduced:
            max_h, max_s, max_l = 255, 63, 63
            mode_text = "Reduced Precision"
        else:
            max_h, max_s, max_l = 359, 99, 99
            mode_text = "Full Precision"
        
        st.markdown(f"**Mode**: {mode_text} - Max possible: H({max_h}), S({max_s}), L({max_l})")
        
        reduction_h = (1 - len(unique_H) / (max_h + 1)) * 100
        reduction_s = (1 - len(unique_S) / (max_s + 1)) * 100
        reduction_l = (1 - len(unique_L) / (max_l + 1)) * 100
        
        st.markdown(f"**Sparsity**: H({reduction_h:.1f}%), S({reduction_s:.1f}%), L({reduction_l:.1f}%) values unused")

        channel_maps = {
            'H': {val: idx for idx, val in enumerate(unique_H)},
            'S': {val: idx for idx, val in enumerate(unique_S)},
            'L': {val: idx for idx, val in enumerate(unique_L)},
        }

        H_ids = np.vectorize(channel_maps['H'].get)(H)
        S_ids = np.vectorize(channel_maps['S'].get)(S)
        L_ids = np.vectorize(channel_maps['L'].get)(L)

        st.subheader("🚀 Encoding Progress")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        debug_table = {}

        def calculate_optimal_bits(value):
            if value == 0:
                return 0
            elif 1 <= value <= 2:
                return 1
            elif 3 <= value <= 6:
                return 2
            elif 7 <= value <= 14:
                return 3
            elif 15 <= value <= 30:
                return 4
            elif 31 <= value <= 62:
                return 5
            elif 63 <= value <= 126:
                return 6
            elif 127 <= value <= 254:
                return 7
            else:
                return 8

        def find_rectangles(channel_data, val):
            h, w = channel_data.shape
            visited = np.zeros_like(channel_data, dtype=bool)
            rectangles = []
            
            for y in range(h):
                for x in range(w):
                    if channel_data[y, x] == val and not visited[y, x]:
                        width = 0
                        while x + width < w and channel_data[y, x + width] == val and not visited[y, x + width]:
                            width += 1
                        
                        height = 1
                        while y + height < h:
                            can_extend = True
                            for dx in range(width):
                                if channel_data[y + height, x + dx] != val or visited[y + height, x + dx]:
                                    can_extend = False
                                    break
                            if not can_extend:
                                break
                            height += 1
                        
                        for dy in range(height):
                            for dx in range(width):
                                visited[y + dy, x + dx] = True
                        
                        rectangles.append((x, y, width, height))
            
            return rectangles

        def encode_channel_stream(channel_data, channel_name, unique_values):
            value_bits_map = {}
            lookup_table_bits = 0
            for val in unique_values:
                bits_needed = calculate_optimal_bits(val)
                value_bits_map[val] = bits_needed
                lookup_table_bits += bits_needed
            compressed_bits = lookup_table_bits
            
            h, w = channel_data.shape
            rect_table = {}
            total = len(unique_values)
            channel_debug = {}
            
            for idx, val in enumerate(unique_values):
                progress_text.text(f"Encoding {channel_name} value {idx + 1}/{total}")
                rectangles = find_rectangles(channel_data, idx)
                rect_table[val] = rectangles
                
                if rectangles:
                    rect_bits = 0
                    rect_details = []
                    last_x, last_y = 0, 0
                    last_w, last_h = 0, 0
                    
                    for i, (rect_x, rect_y, rect_w, rect_h) in enumerate(rectangles):
                        rel_x = rect_x - last_x
                        rel_y = rect_y - last_y
                        
                        if i == 0:
                            delta_w = rect_w
                            delta_h = rect_h
                        else:
                            delta_w = 0 if rect_w == last_w else rect_w
                            delta_h = 0 if rect_h == last_h else rect_h
                        
                        # Split coordinates into separate positive fields
                        if rel_x >= 0:
                            rel_x_pos = rel_x
                            rel_x_neg = 0  # Missing/default
                            x_bits = calculate_optimal_bits(rel_x_pos)
                        else:
                            rel_x_pos = 0  # Missing/default
                            rel_x_neg = abs(rel_x)
                            x_bits = calculate_optimal_bits(rel_x_neg)
                        
                        if rel_y >= 0:
                            rel_y_pos = rel_y
                            rel_y_neg = 0  # Missing/default
                            y_bits = calculate_optimal_bits(rel_y_pos)
                        else:
                            rel_y_pos = 0  # Missing/default
                            rel_y_neg = abs(rel_y)
                            y_bits = calculate_optimal_bits(rel_y_neg)
                        
                        w_bits = calculate_optimal_bits(delta_w)
                        h_bits = calculate_optimal_bits(delta_h)
                        
                        rect_bits += x_bits + y_bits + w_bits + h_bits
                        
                        rect_details.append({
                            'rel_x': rel_x, 'rel_y': rel_y,  # Original values for display
                            'rel_x_pos': rel_x_pos, 'rel_y_pos': rel_y_pos,  # Positive field values
                            'rel_x_neg': rel_x_neg, 'rel_y_neg': rel_y_neg,  # Negative field values
                            'actual_width': rect_w, 'actual_height': rect_h,
                            'encoded_width': delta_w, 'encoded_height': delta_h,
                            'x_bits': x_bits, 'y_bits': y_bits,
                            'w_bits': w_bits, 'h_bits': h_bits,
                            'total_bits': x_bits + y_bits + w_bits + h_bits,
                            'inherited_w': delta_w == 0 and i > 0,
                            'inherited_h': delta_h == 0 and i > 0
                        })
                        
                        last_x = rect_x + rect_w
                        last_y = rect_y
                        last_w = rect_w
                        last_h = rect_h
                    
                    compressed_bits += rect_bits
                    
                    channel_debug[val] = {
                        'rectangles': len(rectangles),
                        'rectangle_coords': rectangles,
                        'rect_bits': rect_bits,
                        'rect_details': rect_details
                    }
                else:
                    channel_debug[val] = {
                        'rectangles': 0,
                        'rectangle_coords': [],
                        'rect_bits': 0,
                        'rect_details': []
                    }
                
                channel_index = {'H': 0, 'S': 1, 'L': 2}[channel_name]
                progress_bar.progress(((idx + 1) + total * channel_index) / (3 * total))
            
            channel_debug['_lookup_table'] = {
                'unique_values': len(unique_values),
                'value_bits_map': value_bits_map,
                'lookup_bits': lookup_table_bits
            }
            debug_table[channel_name] = channel_debug
            return compressed_bits

        bits_H = encode_channel_stream(H_ids, 'H', unique_H)
        bits_S = encode_channel_stream(S_ids, 'S', unique_S)
        bits_L = encode_channel_stream(L_ids, 'L', unique_L)

        compressed_bits = bits_H + bits_S + bits_L
        compressed_bytes = (compressed_bits + 7) // 8
        original_bytes = h * w * 3

        st.session_state.compression_results = {
            'hsl_img': hsl_img,
            'H': H, 'S': S, 'L': L,
            'H_ids': H_ids, 'S_ids': S_ids, 'L_ids': L_ids,
            'unique_H': unique_H, 'unique_S': unique_S, 'unique_L': unique_L,
            'debug_table': debug_table,
            'compressed_bits': compressed_bits,
            'compressed_bytes': compressed_bytes,
            'original_bytes': original_bytes,
            'total_pixels': total_pixels
        }

    results = st.session_state.compression_results
    
    st.subheader("📊 Compression Results")
    st.markdown(f"**Original RGB size:** {results['original_bytes']} bytes")
    st.markdown(f"**Compressed (HSL rectangular format):** {results['compressed_bytes']} bytes")
    st.markdown(f"**Compression ratio:** {results['compressed_bytes']/results['original_bytes']:.2%}")

    if len(results['unique_H']) > 361 or len(results['unique_S']) > 101 or len(results['unique_L']) > 101:
        st.warning("One or more HSL channels have unexpected value ranges.")
    else:
        st.success("Compression completed using per-channel HSL rectangular encoding.")

    st.subheader("🛠️ Compression Analysis")
    
    analysis_data = []
    
    for ch in ['H', 'S', 'L']:
        lookup_info = results['debug_table'][ch]['_lookup_table']
        
        analysis_data.append({
            'Channel': ch,
            'Type': 'Lookup Table',
            'Values': lookup_info['unique_values'],
            'Total Bits': lookup_info['lookup_bits'],
            'Details': f"Unique values: {', '.join([f'{val}({bits}b)' for val, bits in lookup_info['value_bits_map'].items()][:5])}{'...' if len(lookup_info['value_bits_map']) > 5 else ''}"
        })
        
        total_rect_bits = 0
        total_rectangles = 0
        for val, d in results['debug_table'][ch].items():
            if val != '_lookup_table':
                total_rect_bits += d['rect_bits']
                total_rectangles += d['rectangles']
                
                savings_summary = []
                zero_saves = 0
                inherited_saves = 0
                
                for rect in d['rect_details']:
                    if rect['x_bits'] == 0: zero_saves += 1
                    if rect['y_bits'] == 0: zero_saves += 1
                    if rect['w_bits'] == 0 and not rect['inherited_w']: zero_saves += 1
                    if rect['h_bits'] == 0 and not rect['inherited_h']: zero_saves += 1
                    if rect['inherited_w']: inherited_saves += 1
                    if rect['inherited_h']: inherited_saves += 1
                
                if zero_saves > 0: savings_summary.append(f"{zero_saves} zeros")
                if inherited_saves > 0: savings_summary.append(f"{inherited_saves} inherited")
                
                analysis_data.append({
                    'Channel': '',
                    'Type': f'Value {val}',
                    'Values': d['rectangles'],
                    'Total Bits': d['rect_bits'],
                    'Details': f"Rectangles: {d['rectangles']}, Savings: {', '.join(savings_summary) if savings_summary else 'None'}"
                })
        
        analysis_data.append({
            'Channel': f'{ch} Total',
            'Type': 'Summary',
            'Values': total_rectangles,
            'Total Bits': lookup_info['lookup_bits'] + total_rect_bits,
            'Details': f"{lookup_info['lookup_bits']} lookup + {total_rect_bits} rectangles"
        })
        
        analysis_data.append({
            'Channel': '─────',
            'Type': '─────────',
            'Values': '─────',
            'Total Bits': '─────',
            'Details': '─────────────────────────────────────────'
        })
    
    analysis_data = analysis_data[:-1]
    
    import pandas as pd
    df = pd.DataFrame(analysis_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("H Channel", f"{results['debug_table']['H']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in results['debug_table']['H'].items() if k != '_lookup_table')} bits")
    with col2:
        st.metric("S Channel", f"{results['debug_table']['S']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in results['debug_table']['S'].items() if k != '_lookup_table')} bits")
    with col3:
        st.metric("L Channel", f"{results['debug_table']['L']['_lookup_table']['lookup_bits'] + sum(d['rect_bits'] for k, d in results['debug_table']['L'].items() if k != '_lookup_table')} bits")

    st.subheader("📋 Detailed Rectangle Analysis")
    
    rect_data = []
    
    for ch in ['H', 'S', 'L']:
        for val, d in results['debug_table'][ch].items():
            if val != '_lookup_table':
                for i, rect in enumerate(d['rect_details']):
                    savings = []
                    if rect['x_bits'] == 0: savings.append('x=0')
                    if rect['y_bits'] == 0: savings.append('y=0')
                    if rect['inherited_w']: savings.append(f'w=inherited')
                    elif rect['w_bits'] == 0: savings.append('w=0')
                    if rect['inherited_h']: savings.append(f'h=inherited')
                    elif rect['h_bits'] == 0: savings.append('h=0')
                    
                    rect_data.append({
                        'Channel': ch,
                        'Value': val,
                        'Rect #': i + 1,
                        'Rel X': rect['rel_x_pos'] if rect['rel_x_pos'] > 0 else '',
                        'Rel Y': rect['rel_y_pos'] if rect['rel_y_pos'] > 0 else '',
                        'Rel Neg X': rect['rel_x_neg'] if rect['rel_x_neg'] > 0 else '',
                        'Rel Neg Y': rect['rel_y_neg'] if rect['rel_y_neg'] > 0 else '',
                        'Width': rect['actual_width'],
                        'Height': rect['actual_height'],
                        'X Bits': rect['x_bits'],
                        'Y Bits': rect['y_bits'],
                        'W Bits': rect['w_bits'],
                        'H Bits': rect['h_bits'],
                        'Total Bits': rect['total_bits'],
                        'Savings': ', '.join(savings) if savings else 'None',
                        'Area': rect['actual_width'] * rect['actual_height'],
                        'Efficiency': f"{rect['total_bits'] / (rect['actual_width'] * rect['actual_height']):.3f}" if rect['actual_width'] * rect['actual_height'] > 0 else "0"
                    })
    
    rect_df = pd.DataFrame(rect_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        channel_filter = st.selectbox("Filter by Channel:", ['All'] + ['H', 'S', 'L'])
    with col2:
        min_area = st.number_input("Min Area:", min_value=0, value=0)
    with col3:
        show_savings_only = st.checkbox("Show only rectangles with savings")
    
    filtered_df = rect_df.copy()
    if channel_filter != 'All':
        filtered_df = filtered_df[filtered_df['Channel'] == channel_filter]
    if min_area > 0:
        filtered_df = filtered_df[filtered_df['Area'] >= min_area]
    if show_savings_only:
        filtered_df = filtered_df[filtered_df['Savings'] != 'None']
    
    st.dataframe(filtered_df, use_container_width=True, height=500)
    
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rectangles", len(filtered_df))
        with col2:
            st.metric("Avg Bits/Rectangle", f"{filtered_df['Total Bits'].mean():.1f}")
        with col3:
            st.metric("Avg Area", f"{filtered_df['Area'].mean():.1f}")
        with col4:
            savings_count = len(filtered_df[filtered_df['Savings'] != 'None'])
            st.metric("Rectangles with Savings", f"{savings_count} ({100*savings_count/len(filtered_df):.1f}%)")
    else:
        st.info("No rectangles match the current filters.")

    st.subheader("🔄 Test Reconstruction")
    st.write("Reconstruct the image from the compressed HSL rectangular format to verify the compression algorithm.")
    
    if st.button("🎯 Reconstruct Image", type="primary"):
        image = st.session_state.image_data['image']
        img_np = st.session_state.image_data['img_np']
        h = st.session_state.image_data['h']
        w = st.session_state.image_data['w']
        
        progress_reconstruction = st.progress(0)
        reconstruction_text = st.empty()
        
        reconstructed_hsl = np.zeros((h, w, 3), dtype=np.float32)
        
        def reconstruct_channel(channel_data, channel_name, unique_values, channel_index):
            reconstruction_text.text(f"Reconstructing {channel_name} channel...")
            
            for idx, val in enumerate(unique_values):
                if val in results['debug_table'][channel_name] and 'rectangle_coords' in results['debug_table'][channel_name][val]:
                    rectangles = results['debug_table'][channel_name][val]['rectangle_coords']
                    
                    for rect_x, rect_y, rect_w, rect_h in rectangles:
                        if channel_name == 'H':
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 0] = val
                        elif channel_name == 'S':
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 1] = val
                        else:
                            reconstructed_hsl[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, 2] = val
                
                progress_reconstruction.progress((idx + 1 + len(unique_values) * channel_index) / (3 * len(unique_values)))
        
        reconstruct_channel(results['H_ids'], 'H', results['unique_H'], 0)
        reconstruct_channel(results['S_ids'], 'S', results['unique_S'], 1)
        reconstruct_channel(results['L_ids'], 'L', results['unique_L'], 2)
        
        reconstruction_text.text("Converting HSL back to RGB...")
        reconstructed_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        use_reduced = st.session_state.get('reduced_precision', False)
        
        for y in range(h):
            for x in range(w):
                if use_reduced:
                    # Convert back from reduced precision ranges
                    hsl_h = reconstructed_hsl[y, x, 0] / 255.0  # Convert from 0-255 to 0-1
                    hsl_s = reconstructed_hsl[y, x, 1] / 63.0   # Convert from 0-63 to 0-1
                    hsl_l = reconstructed_hsl[y, x, 2] / 63.0   # Convert from 0-63 to 0-1
                else:
                    # Convert back from full precision ranges
                    hsl_h = reconstructed_hsl[y, x, 0] / 360.0  # Convert from 0-359 to 0-1
                    hsl_s = reconstructed_hsl[y, x, 1] / 100.0  # Convert from 0-99 to 0-1
                    hsl_l = reconstructed_hsl[y, x, 2] / 100.0  # Convert from 0-99 to 0-1
                
                rgb_r, rgb_g, rgb_b = colorsys.hls_to_rgb(hsl_h, hsl_l, hsl_s)
                reconstructed_rgb[y, x] = [
                    int(round(rgb_r * 255)),
                    int(round(rgb_g * 255)),
                    int(round(rgb_b * 255))
                ]
            
            if y % 10 == 0:
                progress_reconstruction.progress(0.9 + 0.1 * y / h)
        
        progress_reconstruction.progress(1.0)
        reconstruction_text.text("Reconstruction complete!")
        
        original_hsl_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if use_reduced:
                    hsl_h = results['hsl_img'][y, x, 0] / 255.0
                    hsl_s = results['hsl_img'][y, x, 1] / 63.0
                    hsl_l = results['hsl_img'][y, x, 2] / 63.0
                else:
                    hsl_h = results['hsl_img'][y, x, 0] / 360.0
                    hsl_s = results['hsl_img'][y, x, 1] / 100.0
                    hsl_l = results['hsl_img'][y, x, 2] / 100.0
                
                rgb_r, rgb_g, rgb_b = colorsys.hls_to_rgb(hsl_h, hsl_l, hsl_s)
                original_hsl_rgb[y, x] = [
                    int(round(rgb_r * 255)),
                    int(round(rgb_g * 255)),
                    int(round(rgb_b * 255))
                ]
        
        original_hsl_image = Image.fromarray(original_hsl_rgb)
        reconstructed_image = Image.fromarray(reconstructed_rgb)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original RGB", use_column_width=True)
        with col2:
            st.image(original_hsl_image, caption="Original → HSL → RGB", use_column_width=True)
        with col3:
            st.image(reconstructed_image, caption="Compressed → Reconstructed", use_column_width=True)
        
        hsl_conversion_diff = np.abs(img_np.astype(np.int16) - original_hsl_rgb.astype(np.int16))
        hsl_max_diff = np.max(hsl_conversion_diff)
        hsl_mean_diff = np.mean(hsl_conversion_diff)
        hsl_perfect_pixels = np.sum(np.all(hsl_conversion_diff == 0, axis=2))
        
        compressed_diff = np.abs(img_np.astype(np.int16) - reconstructed_rgb.astype(np.int16))
        compressed_max_diff = np.max(compressed_diff)
        compressed_mean_diff = np.mean(compressed_diff)
        compressed_perfect_pixels = np.sum(np.all(compressed_diff == 0, axis=2))
        
        quantized_hsl = np.zeros_like(results['hsl_img'], dtype=np.float32)
        quantized_hsl[:, :, 0] = results['H'].astype(np.float32)
        quantized_hsl[:, :, 1] = results['S'].astype(np.float32)
        quantized_hsl[:, :, 2] = results['L'].astype(np.float32)
        
        hsl_vs_compressed_hsl_diff = np.abs(quantized_hsl.astype(np.float64) - reconstructed_hsl.astype(np.float64))
        hsl_vs_compressed_hsl_max_diff = np.max(hsl_vs_compressed_hsl_diff)
        hsl_vs_compressed_hsl_mean_diff = np.mean(hsl_vs_compressed_hsl_diff)
        hsl_vs_compressed_hsl_perfect_pixels = np.sum(np.all(hsl_vs_compressed_hsl_diff == 0, axis=2))
        
        st.subheader("📊 Reconstruction Accuracy Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original RGB vs HSL Conversion:**")
            st.metric("Max Difference", f"{hsl_max_diff}")
            st.metric("Mean Difference", f"{hsl_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{hsl_perfect_pixels}/{results['total_pixels']}")
            hsl_accuracy = (hsl_perfect_pixels / results['total_pixels']) * 100
            st.metric("Accuracy", f"{hsl_accuracy:.1f}%")
        
        with col2:
            st.markdown("**Original RGB vs Compressed:**")
            st.metric("Max Difference", f"{compressed_max_diff}")
            st.metric("Mean Difference", f"{compressed_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{compressed_perfect_pixels}/{results['total_pixels']}")
            compressed_accuracy = (compressed_perfect_pixels / results['total_pixels']) * 100
            st.metric("Accuracy", f"{compressed_accuracy:.1f}%")
        
        with col3:
            st.markdown("**Quantized HSL vs Compressed HSL:**")
            st.metric("Max Difference", f"{hsl_vs_compressed_hsl_max_diff:.2f}")
            st.metric("Mean Difference", f"{hsl_vs_compressed_hsl_mean_diff:.2f}")
            st.metric("Perfect Pixels", f"{hsl_vs_compressed_hsl_perfect_pixels}/{results['total_pixels']}")
            hsl_vs_compressed_hsl_accuracy = (hsl_vs_compressed_hsl_perfect_pixels / results['total_pixels']) * 100
            st.metric("Accuracy", f"{hsl_vs_compressed_hsl_accuracy:.1f}%")
        
        st.subheader("🎯 Quality Assessment")
        
        if hsl_vs_compressed_hsl_max_diff == 0:
            st.success("🎉 Perfect compression! The rectangular encoding is perfectly lossless in HSL space.")
        elif hsl_vs_compressed_hsl_max_diff <= 1:
            st.info("✅ Near-perfect compression with minimal differences in HSL space.")
        elif hsl_vs_compressed_hsl_max_diff <= 5:
            st.warning("⚠️ Good compression with small differences in HSL space.")
        else:
            st.error("❌ Significant compression artifacts detected in HSL space.")
        
        if hsl_max_diff > 0 and hsl_vs_compressed_hsl_max_diff == 0:
            st.info("💡 **Analysis**: All precision loss comes from RGB→HSL→RGB conversion. Rectangular compression is perfect!")
        elif hsl_vs_compressed_hsl_max_diff > 0:
            st.warning("⚠️ **Analysis**: The rectangular compression introduces precision loss in HSL space.")
        else:
            st.success("✅ **Analysis**: Perfect preservation in both HSL space and RGB conversion.")
        
        if max(hsl_max_diff, compressed_max_diff, hsl_vs_compressed_hsl_max_diff) > 0:
            st.subheader("🔍 Difference Analysis")
            
            def create_diff_heatmap(pixel_diffs, title):
                pixel_diff_squared = pixel_diffs.astype(np.float64)**2
                diff_sum = np.sum(pixel_diff_squared, axis=2)
                diff_sum = np.maximum(diff_sum, 0)
                diff_magnitude = np.sqrt(diff_sum)
                
                if np.max(diff_magnitude) > 0:
                    diff_heatmap = (diff_magnitude / np.max(diff_magnitude) * 255).astype(np.uint8)
                    heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
                    heatmap_colored[:, :, 0] = diff_heatmap
                    return Image.fromarray(heatmap_colored), diff_magnitude
                return None, diff_magnitude
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if hsl_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(hsl_conversion_diff, "RGB vs HSL")
                    if heatmap_img:
                        st.image(heatmap_img, caption="RGB vs HSL Differences", use_column_width=True)
                else:
                    st.success("No differences in HSL conversion")
            
            with col2:
                if compressed_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(compressed_diff, "RGB vs Compressed")
                    if heatmap_img:
                        st.image(heatmap_img, caption="RGB vs Compressed Differences", use_column_width=True)
                else:
                    st.success("No differences in compression")
            
            with col3:
                if hsl_vs_compressed_hsl_max_diff > 0:
                    heatmap_img, diff_mag = create_diff_heatmap(hsl_vs_compressed_hsl_diff, "HSL vs Compressed HSL")
                    if heatmap_img:
                        st.image(heatmap_img, caption="HSL vs Compressed HSL", use_column_width=True)
                    
                    st.write(f"**Compression-Only Error Statistics:**")
                    st.write(f"- Maximum HSL difference: {np.max(diff_mag):.2f}")
                    st.write(f"- Mean HSL difference: {np.mean(diff_mag):.2f}")
                    st.write(f"- Pixels with differences: {np.sum(diff_mag > 0)} ({np.sum(diff_mag > 0)/results['total_pixels']*100:.1f}%)")
                else:
                    st.success("Perfect rectangular compression!")
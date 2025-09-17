from models import db, Stationnement, ClientToClient, DataRow, FromLastClient, PredictionDetail
from flask import current_app
from sqlalchemy import text
from sqlalchemy import tuple_ as sa_tuple
import statistics
import numpy as np
from datetime import datetime, date, timedelta
import json

class JourneyPredictor:
    """
   
     Simple journey time prediction engine
    Uses historical data from the database to predict journey times
    """
    
    @staticmethod
    def convert_excel_date(excel_date):
        """
        Convert Excel serial date number to Python date object
        Excel stores dates as the number of days since January 1, 1900
        """
        try:
            from datetime import date, timedelta
            
            if isinstance(excel_date, (int, float)):
                # Excel epoch adjustment (Excel incorrectly treats 1900 as leap year)
                excel_epoch = date(1899, 12, 30)
                return excel_epoch + timedelta(days=int(excel_date))
            return None
        except (ValueError, OverflowError):
            return None
    
    def __init__(self):
        self.warehouse_id = 0  # Assuming warehouse has ID 0
        # Simple in-memory caches populated per request to avoid N+1 DB queries
        self._cache = {
            'w2f': {},       # (warehouse, first_client) -> avg minutes
            'c2c': {},       # (from_client, to_client) -> avg minutes
            'l2w': {},       # (last_client, warehouse) -> avg minutes
            'station': {     # stationnement caches
                'by_pallet': {},  # client -> {pallet_count -> avg}
                'overall': {}     # client -> avg
            }
        }

    def preload_caches(self, journeys):
        """
        Preload averages for all needed pairs for a batch of journeys in set-based queries.
        journeys: dict of journey_id -> journey_data
        """
        try:
            if not journeys:
                return

            # Collect all pairs we need
            w2f_pairs = set()
            c2c_pairs = set()
            l2w_pairs = set()
            station_clients = set()
            station_pallets = {}  # client -> set of pallets seen

            for _, j in journeys.items():
                wh = j.get('warehouse')
                stops = j.get('stops', [])
                if not stops:
                    continue
                # warehouse -> first client
                first_client = stops[0].get('store_code')
                if wh and first_client:
                    w2f_pairs.add((wh, first_client))
                # client -> client
                for i in range(len(stops) - 1):
                    a = stops[i].get('store_code')
                    b = stops[i+1].get('store_code')
                    if a and b:
                        c2c_pairs.add((a, b))
                # last client -> warehouse
                last_client = stops[-1].get('store_code')
                if last_client and wh:
                    l2w_pairs.add((last_client, wh))
                # stationnement clients and pallets
                for s in stops:
                    c = s.get('store_code')
                    p = int(s.get('pallets', 0) or 0)
                    if c:
                        station_clients.add(c)
                        station_pallets.setdefault(c, set()).add(p)

            # Preload DataRow (warehouse -> first client)
            if w2f_pairs:
                recs = DataRow.query.filter(
                    sa_tuple(DataRow.FCY0, DataRow.DLVNAM).in_(list(w2f_pairs)),
                    DataRow.ZHCAL.isnot(None)
                ).all()
                # group
                tmp = {}
                for r in recs:
                    key = (r.FCY0, r.DLVNAM)
                    if r.ZHCAL and r.ZHCAL > 0:
                        tmp.setdefault(key, []).append(r.ZHCAL)
                for key, times in tmp.items():
                    filtered = self.remove_outliers_iqr(times)
                    if filtered:
                        self._cache['w2f'][key] = round(sum(filtered) / len(filtered), 2)

            # Preload ClientToClient (between clients)
            if c2c_pairs:
                recs = ClientToClient.query.filter(
                    sa_tuple(ClientToClient.DLVNAM01, ClientToClient.DLVNAM02).in_(list(c2c_pairs)),
                    ClientToClient.TIME.isnot(None)
                ).all()
                tmp = {}
                for r in recs:
                    key = (r.DLVNAM01, r.DLVNAM02)
                    if r.TIME and r.TIME > 0:
                        tmp.setdefault(key, []).append(r.TIME)
                for key, times in tmp.items():
                    filtered = self.remove_outliers_iqr(times)
                    if filtered:
                        self._cache['c2c'][key] = round(sum(filtered) / len(filtered), 2)

            # Preload FromLastClient (last -> warehouse)
            if l2w_pairs:
                recs = FromLastClient.query.filter(
                    sa_tuple(FromLastClient.DLVNAM0, FromLastClient.FCY0).in_(list(l2w_pairs)),
                    FromLastClient.time.isnot(None)
                ).all()
                tmp = {}
                for r in recs:
                    key = (r.DLVNAM0, r.FCY0)
                    if r.time and r.time > 0:
                        tmp.setdefault(key, []).append(r.time)
                for key, times in tmp.items():
                    filtered = self.remove_outliers_iqr(times)
                    if filtered:
                        self._cache['l2w'][key] = round(sum(filtered) / len(filtered), 2)

            # Preload Stationnement per client (+ by pallet buckets)
            if station_clients:
                recs = Stationnement.query.filter(
                    Stationnement.DLVNAM0.in_(list(station_clients)),
                    Stationnement.time.isnot(None)
                ).all()
                by_client = {}
                for r in recs:
                    c = r.DLVNAM0
                    t = r.time
                    if not t or t <= 0:
                        continue
                    pallets_val = getattr(r, 'SHPNBR0', getattr(r, 'SHPNBR', 0)) or 0
                    by_client.setdefault(c, []).append({'time': t, 'SHPNBR': pallets_val})
                for c, rows in by_client.items():
                    # overall
                    all_times = [x['time'] for x in rows]
                    filtered = self.remove_outliers_iqr(all_times)
                    if filtered:
                        self._cache['station']['overall'][c] = round(sum(filtered) / len(filtered), 2)
                    # by pallet buckets (only for pallets we actually need)
                    pallets_needed = station_pallets.get(c, set())
                    for p in pallets_needed:
                        times_p = [x['time'] for x in rows if (x['SHPNBR'] or 0) == p]
                        if times_p:
                            filt_p = self.remove_outliers_iqr(times_p)
                            if filt_p:
                                self._cache['station']['by_pallet'].setdefault(c, {})[p] = round(sum(filt_p)/len(filt_p), 2)
        except Exception as e:
            # Cache preload is best-effort; on error we continue with on-demand queries
            print(f"Cache preload warning: {e}")
    
    def extract_journey_details(self, sql_data):
      
      
        journeys = {}
        
        # Convert SQL result to list of dictionaries if needed
        if hasattr(sql_data, 'fetchall'):
            # If it's a SQL result object
            sql_data = [dict(row) for row in sql_data]
        elif hasattr(sql_data[0], '__dict__') if sql_data else False:
            # If it's SQLAlchemy model objects
            sql_data = [self._model_to_dict(row) for row in sql_data]
        
        # Group data by journey identifier (QYMNUM_0)
        for row in sql_data:
            journey_id = row.get('QYMNUM_0') or row.get('QYMNUM0')
            
            if journey_id is None:
                print(f"Skipping row - no journey ID found. Row keys: {list(row.keys())}")
                continue
            
            # Initialize journey if not exists
            if journey_id not in journeys:
                # Get and convert date
                raw_date = row.get('DAT_0') or row.get('DAT0', '')
                converted_date = self.convert_excel_date(raw_date) if isinstance(raw_date, (int, float)) else raw_date
                # Extract starting time from TIM_0
                starting_time = row.get('TIM_0') or row.get('TIM0', 0.0)
                journeys[journey_id] = {
                    'warehouse': row.get('FCY_0') or row.get('FCY0'),
                    'date': converted_date.isoformat() if hasattr(converted_date, 'isoformat') else str(converted_date) if converted_date else '',
                    'starting_time': starting_time,
                    'stops': [],
                    'total_pallets': 0,
                    'stop_count': 0
                }
            
            # Extract stop details
            stop_data = {
                'order': row.get('ZHARCL_0') or row.get('ZHARCL0', 0),
                'store_code': row.get('DLVNAM_0') or row.get('DLVNAM0'),
                'store_name': row.get('DLVNAM_1') or row.get('DLVNAM1', ''),
                'pallets': row.get('SHPNBR_0') or row.get('SHPNBR0', 0)
            }
            
            journeys[journey_id]['stops'].append(stop_data)
        
        print(f"Extracted {len(journeys)} journeys from {len(sql_data)} rows")
        
        # Process each journey: sort stops and calculate totals
        for journey_id, journey_data in journeys.items():
            # Sort stops by ZHARCL_0 in ascending order (smaller to bigger)
            def safe_order_value(val):
                if val is None:
                    return 0
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    val = val.strip()
                    if val == '' or not val.isdigit():
                        return 0
                    return int(val)
                return 0
            journey_data['stops'].sort(key=lambda x: safe_order_value(x['order']))
            
            # Calculate totals
            journey_data['total_pallets'] = sum(stop['pallets'] for stop in journey_data['stops'])
            journey_data['stop_count'] = len(journey_data['stops'])
            
            # Add stop sequence numbers for easier reference
            for i, stop in enumerate(journey_data['stops']):
                stop['sequence'] = i + 1
        print("journeys",journeys)
        return journeys
        
    
    def _model_to_dict(self, model_obj):
        """
        Convert SQLAlchemy model object to dictionary
        """
        if hasattr(model_obj, '__dict__'):
            return {key: value for key, value in model_obj.__dict__.items() 
                   if not key.startswith('_')}
        return model_obj
    
    def get_journey_route(self, journey_data):
        """
        Extract the route (warehouse -> stops -> warehouse) from journey data
        
        Args:
            journey_data: Single journey data from extract_journey_details
        
        Returns:
            List of store codes representing the complete route
        """
        if not journey_data or not journey_data.get('stops'):
            return []
        
        # Start with warehouse
        route = [journey_data['warehouse']]
        
        # Add all stops in order
        for stop in journey_data['stops']:
            route.append(stop['store_code'])
        
        # Return to warehouse
        route.append(journey_data['warehouse'])
        
        return route
    
    def analyze_journeys_from_table(self, table_name_or_query):
        """
        Analyze journeys directly from database table or custom query
        
        Args:
            table_name_or_query: Either table name string or SQL query string
        
        Returns:
            Dictionary with journey analysis
        """
        try:
            if table_name_or_query.upper().startswith('SELECT'):
                # It's a custom query
                result = db.session.execute(text(table_name_or_query))
            else:
                # It's a table name
                result = db.session.execute(text(f"SELECT * FROM {table_name_or_query}"))
            
            # Convert SQL result to list of dictionaries - handle different SQLAlchemy versions
            sql_data = []
            rows = result.fetchall()
            
            if rows:
                # Get column names
                if hasattr(result, 'keys'):
                    columns = result.keys()
                else:
                    columns = rows[0]._fields if hasattr(rows[0], '_fields') else []
                
                # Convert each row
                for row in rows:
                    if hasattr(row, '_asdict'):
                        # Named tuple approach
                        sql_data.append(row._asdict())
                    elif hasattr(row, '_mapping'):
                        # SQLAlchemy 1.4+ approach
                        sql_data.append(dict(row._mapping))
                    else:
                        # Manual conversion
                        row_dict = {}
                        for i, col in enumerate(columns):
                            row_dict[col] = row[i] if i < len(row) else None
                        sql_data.append(row_dict)
            
            # Extract journey details
            journeys = self.extract_journey_details(sql_data)
            
            # Analyze each journey
            journey_analysis = {}
            for journey_id, journey_data in journeys.items():
                route = self.get_journey_route(journey_data)
                print("journey_data", journey_data)
                predicted_result = self.predict_complete_journey_time(journey_data, journey_id)
                
                journey_analysis[journey_id] = {
                    'journey_data': journey_data,
                    'route': route,
                    'prediction_result': predicted_result,
                    'route_summary': f"{journey_data['warehouse']} -> {' -> '.join([str(stop['store_code']) for stop in journey_data['stops']])} -> {journey_data['warehouse']}"
                }
            
            return journey_analysis
            
        except Exception as e:
            print(f"Error analyzing journeys: {e}")
            return {}
    
    def calculate_warehouse_to_first_client_time(self, warehouse_code, first_client_code):
        """
        Function 1: Calculate average time from warehouse to first client
        Uses table: to_first_client (DataRow model)
        
        Args:
            warehouse_code: Warehouse identifier from FCY0
            first_client_code: First stop identifier from DLVNAM
        
        Returns:
            Average time in minutes, None if no data
        """
        try:
            # Fast path: cached
            cached = self._cache.get('w2f', {}).get((warehouse_code, first_client_code))
            if cached is not None:
                return cached
            # Query to_first_client table for matching warehouse and client
            records = DataRow.query.filter(
                DataRow.FCY0 == warehouse_code,
                DataRow.DLVNAM == first_client_code,
                DataRow.ZHCAL.isnot(None)
            ).all()
            
            if not records:
                print(f"No data found for warehouse {warehouse_code} to client {first_client_code}")
                return None
            
            # Extract times and filter positive values
            times = [r.ZHCAL for r in records if r.ZHCAL and r.ZHCAL > 0]
            
            if not times:
                return None
            
            # Remove outliers using IQR method
            filtered_times = self.remove_outliers_iqr(times)
            
            if not filtered_times:
                return None
            
            # Calculate average (mean)
            average_time = sum(filtered_times) / len(filtered_times)
            
            print(f"Warehouse to first client: {len(times)} records, {len(filtered_times)} after outlier removal")
            
            return round(average_time, 2)
            
        except Exception as e:
            print(f"Error calculating warehouse to first client time: {e}")
            return None
    
    def calculate_client_to_client_times(self, client_stops):
        """
        Function 2: Calculate average times between consecutive clients
        Uses table: client_to_client (ClientToClient model)

        Args:
            client_stops: List of client codes in order [client1, client2, client3, ...]

        Returns:
            List of average times for each segment, None for segments with no data
        """
        if len(client_stops) < 2:
            return []

        segment_times = []

        for i in range(len(client_stops) - 1):
            from_client = str(client_stops[i])     # ensure string because columns are TEXT
            to_client = str(client_stops[i + 1])   # ensure string because columns are TEXT

            try:
                # Use cache if available
                cached = self._cache.get('c2c', {}).get((from_client, to_client))
                if cached is not None:
                    segment_times.append(cached)
                    print(f"Client {from_client} to {to_client}: cached average {cached}")
                    continue

                # Query client_to_client table for this segment
                records = ClientToClient.query.filter(
                    ClientToClient.DLVNAM_01 == from_client,
                    ClientToClient.DLVNAM_02 == to_client,
                    ClientToClient.time_minutes.isnot(None)
                ).all()

                if not records:
                    print(f"No data found for client {from_client} to client {to_client}")
                    segment_times.append(None)
                    continue

                # Extract valid positive times
                times = [r.time_minutes for r in records if r.time_minutes and r.time_minutes > 0]

                if not times:
                    segment_times.append(None)
                    continue

                # Remove outliers (IQR method)
                filtered_times = self.remove_outliers_iqr(times)

                if not filtered_times:
                    segment_times.append(None)
                    continue

                # Calculate average (mean)
                average_time = sum(filtered_times) / len(filtered_times)

                print(
                    f"Client {from_client} to {to_client}: "
                    f"{len(times)} records, {len(filtered_times)} after outlier removal"
                )

                segment_times.append(round(average_time, 2))

            except Exception as e:
                print(f"Error calculating client to client time ({from_client} -> {to_client}): {e}")
                segment_times.append(None)

        return segment_times

    
    def calculate_last_client_to_warehouse_time(self, last_client_code, warehouse_code):
        """
        Function 3: Calculate average time from last client back to warehouse
        Uses table: from_last_client (FromLastClient model)
        
        Args:
            last_client_code: Last stop identifier
            warehouse_code: Warehouse identifier
        
        Returns:
            Average time in minutes, None if no data
        """
        try:
            cached = self._cache.get('l2w', {}).get((last_client_code, warehouse_code))
            if cached is not None:
                return cached
            # Query from_last_client table
            records = FromLastClient.query.filter(
                FromLastClient.DLVNAM0 == last_client_code,
                FromLastClient.FCY0 == warehouse_code,
                FromLastClient.time.isnot(None)
            ).all()
            
            if not records:
                print(f"No data found for last client {last_client_code} to warehouse {warehouse_code}")
                return None
            
            # Extract times and filter positive values
            times = [r.time for r in records if r.time and r.time > 0]
            
            if not times:
                return None
            
            # Remove outliers using IQR method
            filtered_times = self.remove_outliers_iqr(times)
            
            if not filtered_times:
                return None
            
            # Calculate average (mean)
            average_time = sum(filtered_times) / len(filtered_times)
            
            print(f"Last client to warehouse: {len(times)} records, {len(filtered_times)} after outlier removal")
            
            return round(average_time, 2)
            
        except Exception as e:
            print(f"Error calculating last client to warehouse time: {e}")
            return None
    
    def calculate_stationnement_times(self, client_stops):
        """
        Function 4: Calculate average stationnement (stop) time for each client
        Uses table: stationnement (Stationnement model)
        Takes into consideration the number of pallets (SHPNBR)
        
        Args:
            client_stops: List of client codes with pallet information
                         Expected format: [{'store_code': 123, 'pallets': 5}, ...]
        
        Returns:
            List of average stop times for each client, None for clients with no data
        """
        stationnement_times = []
        
        # Use cache if available for fast path
        station_cache = self._cache.get('station', {})
        has_cache = bool(station_cache.get('overall'))
        for stop_info in client_stops:
            # Handle both formats: simple codes or stop dictionaries
            if isinstance(stop_info, dict):
                client_code = stop_info.get('store_code')
                pallet_count = stop_info.get('pallets', 0)
            else:
                client_code = stop_info
                pallet_count = 0
            if has_cache and client_code in station_cache.get('overall', {}):
                by_p = station_cache.get('by_pallet', {}).get(client_code, {})
                if pallet_count in by_p:
                    stationnement_times.append(by_p[pallet_count])
                    continue
                stationnement_times.append(station_cache['overall'][client_code])
                continue
            try:
                # Query stationnement table for this client
                records = Stationnement.query.filter(
                    Stationnement.DLVNAM0 == client_code,
                    Stationnement.time.isnot(None)
                ).all()
                
                if not records:
                    print(f"No stationnement data found for client {client_code}")
                    stationnement_times.append(None)
                    continue
                
                # Convert to list of dictionaries for analysis
                data_list = []
                for record in records:
                    if record.time and record.time > 0:
                        data_list.append({
                            'time': record.time,
                            'SHPNBR': getattr(record, 'SHPNBR0', getattr(record, 'SHPNBR', 0))
                        })
                
                if not data_list:
                    stationnement_times.append(None)
                    continue
                
                # If we have pallet information, try to find matching pallet count
                if pallet_count > 0:
                    # First try to find records with exact pallet match
                    exact_matches = [d for d in data_list if d['SHPNBR'] == pallet_count]
                    
                    if exact_matches:
                        times = [d['time'] for d in exact_matches]
                        filtered_times = self.remove_outliers_iqr(times)
                        if filtered_times:
                            average_time = sum(filtered_times) / len(filtered_times)
                            print(f"Stationnement for client {client_code} with {pallet_count} pallets: {len(times)} records, {len(filtered_times)} after outlier removal")
                            stationnement_times.append(round(average_time, 2))
                            continue
                
                # If no exact pallet match or no pallet info, use pallet-based analysis
                predicted_time = self._calculate_pallet_based_stationnement(data_list, pallet_count)
                
                if predicted_time is not None:
                    print(f"Stationnement for client {client_code}: pallet-based prediction = {predicted_time}")
                    stationnement_times.append(predicted_time)
                else:
                    # Fallback to general average
                    all_times = [d['time'] for d in data_list]
                    filtered_times = self.remove_outliers_iqr(all_times)
                    if filtered_times:
                        average_time = sum(filtered_times) / len(filtered_times)
                        print(f"Stationnement for client {client_code}: general average = {average_time}")
                        stationnement_times.append(round(average_time, 2))
                    else:
                        stationnement_times.append(None)
                
            except Exception as e:
                print(f"Error calculating stationnement time for client {client_code}: {e}")
                stationnement_times.append(None)
        
        return stationnement_times
    
    def _calculate_pallet_based_stationnement(self, data_list, target_pallets):
        """
        Calculate stationnement time based on pallet analysis
        
        Args:
            data_list: List of {'time': x, 'SHPNBR': y} dictionaries
            target_pallets: Number of pallets for prediction
        
        Returns:
            Predicted time or None
        """
        try:
            # Group by pallet count and calculate stats
            pallet_groups = {}
            for record in data_list:
                pallet_count = record['SHPNBR']
                if pallet_count not in pallet_groups:
                    pallet_groups[pallet_count] = []
                pallet_groups[pallet_count].append(record['time'])
            
            # Create pallet statistics similar to your function
            pallet_stats = []
            for pallet_count in sorted(pallet_groups.keys()):
                times = pallet_groups[pallet_count]
                
                # Remove outliers for this pallet count using IQR
                filtered_times = self.remove_outliers_iqr(times)
                
                if filtered_times:
                    avg_time = sum(filtered_times) / len(filtered_times)
                    pallet_stats.append({
                        "SHPNBR": pallet_count,
                        "average_time": avg_time,
                        "count": len(filtered_times)
                    })
            
            if not pallet_stats:
                return None
            
            # If we have target pallets, try to predict based on pattern
            if target_pallets > 0 and len(pallet_stats) >= 2:
                # Find closest pallet counts
                lower_stat = None
                upper_stat = None
                
                for stat in pallet_stats:
                    if stat["SHPNBR"] <= target_pallets:
                        lower_stat = stat
                    elif stat["SHPNBR"] > target_pallets and upper_stat is None:
                        upper_stat = stat
                        break
                
                # Interpolate if we have bounds
                if lower_stat and upper_stat and lower_stat["SHPNBR"] != upper_stat["SHPNBR"]:
                    # Linear interpolation
                    x1, y1 = lower_stat["SHPNBR"], lower_stat["average_time"]
                    x2, y2 = upper_stat["SHPNBR"], upper_stat["average_time"]
                    predicted = y1 + (y2 - y1) * (target_pallets - x1) / (x2 - x1)
                    return round(predicted, 2)
                
                # If exact match or only one bound, use closest
                if lower_stat:
                    return round(lower_stat["average_time"], 2)
                if upper_stat:
                    return round(upper_stat["average_time"], 2)
            
            # Fallback: use overall average
            if pallet_stats:
                total_time = sum(stat["average_time"] * stat["count"] for stat in pallet_stats)
                total_count = sum(stat["count"] for stat in pallet_stats)
                return round(total_time / total_count, 2) if total_count > 0 else None
            
            return None
            
        except Exception as e:
            print(f"Error in pallet-based calculation: {e}")
            return None
    
    def predict_complete_journey_time(self, journey_data, journey_id=None):
        """
        Complete journey time prediction using all 4 specialized functions
        
        Args:
            journey_data: Journey data from extract_journey_details
            journey_id: Optional journey identifier for logging
        
        Returns:
            Dictionary with detailed time breakdown
        """
        if not journey_data or not journey_data.get('stops'):
            return {"error": "Invalid journey data"}
        
        warehouse_code = journey_data['warehouse']
        client_stops = [stop['store_code'] for stop in journey_data['stops']]
        
        if not client_stops:
            return {"error": "No client stops found"}
        
        # Check for missing client-to-client data first
        missing_segments = self._check_missing_client_to_client_data(client_stops)
        if missing_segments:
            # Log to missing_time file and return error
            self._log_missing_data(journey_data, missing_segments, journey_id)
            missing_info = "; ".join([f"no data between {seg['from']} and {seg['to']}" for seg in missing_segments])
            return {
                "error": f"Missing client-to-client data: {missing_info}",
                "has_missing_data": True,  # Flag to exclude from database saving
                "missing_segments": missing_segments
            }
        
        result = {
            "warehouse": warehouse_code,
            "stops": client_stops,
            "time_breakdown": {},
            "total_time": 0,
            "segments_calculated": 0,
            "segments_with_data": 0
        }
        
        # Function 1: Warehouse to first client
        first_client = client_stops[0]
        warehouse_to_first = self.calculate_warehouse_to_first_client_time(warehouse_code, first_client)
        result["time_breakdown"]["warehouse_to_first_client"] = {
            "from": warehouse_code,
            "to": first_client,
            "time": warehouse_to_first
        }
        
        # Function 2: Client to client times
        client_to_client_times = self.calculate_client_to_client_times(client_stops)
        result["time_breakdown"]["client_to_client"] = []
        
        for i, time_value in enumerate(client_to_client_times):
            segment = {
                "from": client_stops[i],
                "to": client_stops[i + 1],
                "time": time_value
            }
            result["time_breakdown"]["client_to_client"].append(segment)
        
        # Function 3: Last client to warehouse
        last_client = client_stops[-1]
        last_to_warehouse = self.calculate_last_client_to_warehouse_time(last_client, warehouse_code)
        result["time_breakdown"]["last_client_to_warehouse"] = {
            "from": last_client,
            "to": warehouse_code,
            "time": last_to_warehouse
        }
        
        # Function 4: Stationnement times (pass complete stop info including pallets)
        stationnement_times = self.calculate_stationnement_times(journey_data['stops'])
        result["time_breakdown"]["stationnement"] = []
        
        for i, time_value in enumerate(stationnement_times):
            stop_data = {
                "client": client_stops[i],
                "pallets": journey_data['stops'][i]['pallets'],
                "stationnement_time": time_value
            }
            result["time_breakdown"]["stationnement"].append(stop_data)
        
        # Calculate total time
        total_time = 0
        segments_with_data = 0
        total_segments = 0
        
        # Add warehouse to first client time
        if warehouse_to_first is not None:
            total_time += warehouse_to_first
            segments_with_data += 1
        total_segments += 1
        
        # Add client to client times
        for time_value in client_to_client_times:
            if time_value is not None:
                total_time += time_value
                segments_with_data += 1
            total_segments += 1
        
        # Add last client to warehouse time
        if last_to_warehouse is not None:
            total_time += last_to_warehouse
            segments_with_data += 1
        total_segments += 1
        
        # Add stationnement times
        for time_value in stationnement_times:
            if time_value is not None:
                total_time += time_value
                segments_with_data += 1
            total_segments += 1
        
        result["total_time"] = round(total_time, 2)
        result["segments_calculated"] = total_segments
        result["segments_with_data"] = segments_with_data
        result["data_coverage_percentage"] = round((segments_with_data / total_segments) * 100, 2) if total_segments > 0 else 0
        
        return result
    
    def _check_missing_client_to_client_data(self, client_stops):
        """
        Check if there are missing client-to-client segments
        
        Args:
            client_stops: List of client codes in order
        
        Returns:
            List of missing segments with 'from' and 'to' client codes
        """
        if len(client_stops) < 2:
            return []

        missing_segments = []

        for i in range(len(client_stops) - 1):
            from_client = client_stops[i]
            to_client = client_stops[i + 1]
            
            try:
                # Check cache first
                cached = self._cache.get('c2c', {}).get((from_client, to_client))
                if cached is not None:
                    continue
                    
                # Query database for this segment
                records = ClientToClient.query.filter(
                    ClientToClient.DLVNAM01 == from_client,
                    ClientToClient.DLVNAM02 == to_client,
                    ClientToClient.TIME.isnot(None)
                ).all()
                
                # Check if we have valid data
                valid_times = [r.TIME for r in records if r.TIME and r.TIME > 0]
                
                if not valid_times:
                    missing_segments.append({
                        'from': from_client,
                        'to': to_client
                    })
                    
            except Exception as e:
                print(f"Error checking client-to-client data ({from_client} -> {to_client}): {e}")
                # Treat errors as missing data
                missing_segments.append({
                    'from': from_client,
                    'to': to_client
                })

        return missing_segments
    
    def _log_missing_data(self, journey_data, missing_segments, journey_id=None):
        """
        Log journey with missing data to missing_time file
        
        Args:
            journey_data: Journey data from extract_journey_details
            missing_segments: List of missing segments
            journey_id: Journey identifier
        """
        try:
            import os
            from datetime import datetime
            
            # Create missing_time file if it doesn't exist
            missing_file = 'missing_time.txt'
            
            # Prepare log entry
            journey_id = journey_id or journey_data.get('QYMNUM_0', 'Unknown')
            warehouse = journey_data.get('warehouse', 'Unknown')
            date = journey_data.get('date', 'Unknown')
            
            missing_info = []
            for segment in missing_segments:
                missing_info.append(f"no data between {segment['from']} and {segment['to']}")
            
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] QYMNUM: {journey_id} | Warehouse: {warehouse} | Date: {date} | {'; '.join(missing_info)}\n"
            
            # Append to file
            with open(missing_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print(f"📝 Logged missing data for journey {journey_id} to {missing_file}")
            
        except Exception as e:
            print(f"❌ Error logging missing data: {e}")

    def remove_outliers_iqr(self, data, multiplier=1.5):
        """
        Remove outliers using the IQR (Interquartile Range) method
        
        Args:
            data: List of numerical values
            multiplier: IQR multiplier (default 1.5 for standard outlier detection)
        
        Returns:
            List of values with outliers removed
        """
        if not data or len(data) < 4:  # Need at least 4 points for meaningful IQR
            return data
        
        # Convert to numpy array for easier calculations
        arr = np.array(data)
        
        # Calculate quartiles
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filter outliers
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        
        # Return original data if filtering removes too many points (>50%)
        if len(filtered_data) < len(data) * 0.5:
            return data
        
        return filtered_data
    
    def get_outlier_stats(self, data, multiplier=1.5):
        """
        Get statistics about outlier detection for debugging
        
        Args:
            data: List of numerical values
            multiplier: IQR multiplier
        
        Returns:
            Dictionary with outlier statistics
        """
        if not data or len(data) < 4:
            return {
                "original_count": len(data) if data else 0,
                "filtered_count": len(data) if data else 0,
                "outliers_removed": 0,
                "outlier_percentage": 0,
                "bounds": {"lower": None, "upper": None}
            }
        
        arr = np.array(data)
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        outliers_removed = len(data) - len(filtered_data)
        
        return {
            "original_count": len(data),
            "filtered_count": len(filtered_data),
            "outliers_removed": outliers_removed,
            "outlier_percentage": round((outliers_removed / len(data)) * 100, 2),
            "bounds": {"lower": round(lower_bound, 2), "upper": round(upper_bound, 2)},
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2)
        }
    
    def save_prediction_normalized(self, journey_id, journey_data, prediction_result):
        """
        Save prediction result to the database using normalized structure
        Creates multiple rows (one per stop) like the original data format
        """
        try:
            # Safety check: Do not save journeys with missing data
            if prediction_result.get("has_missing_data", False):
                print(f"⚠️ Skipping database save for journey {journey_id} - has missing client-to-client data")
                return None
            
            # Safety check: Do not save journeys with errors
            if prediction_result.get("error"):
                print(f"⚠️ Skipping database save for journey {journey_id} - has prediction error")
                return None
                
            from models import PredictionDetail
            import json
            from datetime import datetime
            
            # Extract basic journey information
            warehouse_code = journey_data.get('warehouse', '')
            stops = journey_data.get('stops', [])
            date_str = journey_data.get('date', '')
            
            # Convert date string to date object
            prediction_date = None
            if date_str:
                try:
                    if isinstance(date_str, str) and date_str:
                        prediction_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    elif isinstance(date_str, (int, float)) and date_str > 0:
                        # Excel date conversion
                        excel_date = int(date_str)
                        base_date = datetime(1899, 12, 30)
                        prediction_date = (base_date + timedelta(days=excel_date)).date()
                except (ValueError, TypeError) as e:
                    print(f"Date conversion error: {e}")
                    prediction_date = None
            
            # Extract prediction data
            total_time = prediction_result.get('total_time', 0.0)
            time_breakdown = prediction_result.get('time_breakdown', {})
            
            # Get starting time (TIM_0) from journey_data - format HHMM
            starting_time = journey_data.get('starting_time', 0.0)
            
            # Helper function to add minutes to HHMM format
            def add_minutes_to_hhmm(hhmm_time, minutes_to_add):
                """
                Add minutes to time in HHMM format
                Example: add_minutes_to_hhmm(815, 60) returns 915 (08:15 + 60min = 09:15)
                """
                # Handle None values for minutes_to_add
                if minutes_to_add is None:
                    minutes_to_add = 0
                if hhmm_time is None:
                    hhmm_time = 0
                if minutes_to_add == 0:
                    return hhmm_time
                # Convert HHMM to total minutes since midnight
                hours = int(hhmm_time) // 100
                mins = int(hhmm_time) % 100
                total_minutes = hours * 60 + mins
                # Add the predicted minutes
                total_minutes += int(minutes_to_add)
                # Handle day overflow (if time goes past 24:00)
                total_minutes = total_minutes % (24 * 60)
                # Convert back to HHMM format
                new_hours = total_minutes // 60
                new_mins = total_minutes % 60
                return new_hours * 100 + new_mins
            
            # Calculate return time (ZHRD_0) = starting_time + total_predicted_time (in HHMM format)
            return_time = add_minutes_to_hhmm(starting_time, total_time)
            
            # Get individual segment times
            warehouse_to_first = time_breakdown.get('warehouse_to_first_client', {}).get('time', 0.0) or 0.0
            client_to_client_times = time_breakdown.get('client_to_client', [])
            stationnement_times = time_breakdown.get('stationnement', [])
            last_to_warehouse = time_breakdown.get('last_client_to_warehouse', {}).get('time', 0.0) or 0.0
            
            # Track current departure time for next iteration
            current_departure_time = starting_time
            
            saved_records = []
            
            for i, stop in enumerate(stops):
                # Extract stop information
                client_code = str(stop.get('store_code', '') or stop.get('client_code', ''))
                client_name = str(stop.get('store_name', '') or stop.get('client_name', ''))
                pallets = int(stop.get('pallets', 0))
                
                # Calculate arrival time (ZHARCL_0)
                if i == 0:
                    # First client: ZHARCL = TIM_0 + time predicted (to_first_client)
                    arrival_time = add_minutes_to_hhmm(starting_time, warehouse_to_first)
                else:
                    # Subsequent clients: ZHARCL = ZHDECL(previous) + time predicted(client_to_client)
                    if i-1 < len(client_to_client_times):
                        travel_time = client_to_client_times[i-1].get('time', 0) if isinstance(client_to_client_times[i-1], dict) else client_to_client_times[i-1] or 0
                        arrival_time = add_minutes_to_hhmm(current_departure_time, travel_time)
                    else:
                        arrival_time = current_departure_time
                
                # Calculate stationnement time for this stop
                stationnement_time = 0
                if i < len(stationnement_times) and stationnement_times[i]:
                    stationnement_time = stationnement_times[i].get('stationnement_time', 0) if isinstance(stationnement_times[i], dict) else stationnement_times[i] or 0
                
                # Calculate departure time: ZHDECL_0 = ZHARCL_0 + time predicted (stationnement)
                departure_time = add_minutes_to_hhmm(arrival_time, stationnement_time)
                
                # Determine ZHRD_0 (return time) - only set on last row
                journey_return_time = return_time if i == len(stops) - 1 else None
                
                # Create prediction record for this stop
                prediction_record = PredictionDetail(
                    FCY_0=warehouse_code,
                    DAT_0=prediction_date,
                    QYMNUM_0=str(journey_id),
                    TIM_0=starting_time,
                    ZHRD_0=journey_return_time,
                    DLVNAM_0=client_code,
                    SHPNBR=pallets,
                    ZHARCL_0=arrival_time,
                    ZHDECL_0=departure_time,
                    DLVNAM_1=client_name,
                    is_prediction=True,
                    data_source='prediction_engine'
                )
                
                # Collect for bulk insert
                saved_records.append(prediction_record)
                
                # Update current_departure_time for next iteration
                current_departure_time = departure_time
                
                print(f"✅ Created prediction record for stop {i+1}: {client_code} ({client_name})")
            
            # Bulk insert for better performance on many rows
            if saved_records:
                db.session.bulk_save_objects(saved_records)
                db.session.commit()
            
            print(f"✅ Successfully saved {len(saved_records)} prediction records for journey {journey_id}")
            return len(saved_records)
            
        except Exception as e:
            print(f"❌ Error saving prediction: {str(e)}")
            db.session.rollback()
            import traceback
            traceback.print_exc()
            return None

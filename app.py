from flask import Flask, request, jsonify, render_template
from sqlalchemy import text
from dotenv import load_dotenv
import os
import json
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Import models and initialize database
from models import db, Stationnement, ClientToClient, DataRow, FromLastClient
db.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    try:
        # Test database connection
        db.session.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500

@app.route('/debug', methods=['POST'])
def debug():
    """
    Debug endpoint to see raw data from table/query
    """
    try:
        from prediction_engine import JourneyPredictor
        from sqlalchemy import text
        
        data = request.get_json()
        table_or_query = data.get('table_or_query', '')
        
        if not table_or_query:
            return {"error": "No table name or query provided"}, 400
        
        # Get raw data from database
        if table_or_query.upper().startswith('SELECT'):
            result = db.session.execute(text(table_or_query))
        else:
            result = db.session.execute(text(f"SELECT * FROM {table_or_query} LIMIT 5"))
        
        # Convert to list of dictionaries - handle different SQLAlchemy versions
        raw_data = []
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
                    raw_data.append(row._asdict())
                elif hasattr(row, '_mapping'):
                    # SQLAlchemy 1.4+ approach
                    raw_data.append(dict(row._mapping))
                else:
                    # Manual conversion
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i] if i < len(row) else None
                    raw_data.append(row_dict)
        
        return {
            "raw_data": raw_data,
            "row_count": len(raw_data),
            "status": "success",
            "first_row_keys": list(raw_data[0].keys()) if raw_data else [],
            "sample_qymnum_values": [row.get('QYMNUM_0') for row in raw_data[:3]] if raw_data else [],
            "has_qymnum_0": any('QYMNUM_0' in row for row in raw_data) if raw_data else False
        }
    
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}, 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        from prediction_engine import JourneyPredictor
        from datetime import datetime
        
        data = request.get_json()
        predictor = JourneyPredictor()
        
        # Get date range from request
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        
        if not date_from or not date_to:
            return {"error": "Both date_from and date_to are required (YYYY-MM-DD format)"}, 400
        
        # Validate date format
        try:
            datetime.strptime(date_from, '%Y-%m-%d')
            datetime.strptime(date_to, '%Y-%m-%d')
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}, 400
        
        # Build query for sql_data_backup table with date range
        query = f"SELECT * FROM sql_data_backup WHERE DAT_0 BETWEEN '{date_from}' AND '{date_to}'"
        
        # Get journeys from database using the constructed query
        analysis = predictor.analyze_journeys_from_table(query)
        
        # Extract just the journey data for prediction
        journeys = {}
        for journey_id, analysis_data in analysis.items():
            journeys[journey_id] = analysis_data['journey_data']
        
        if not journeys:
            return {"error": "No valid journeys found in the data"}, 400
        
        # Preload caches to avoid N+1 queries per journey
        print(f"📊 Found {len(journeys)} journeys in date range {date_from} to {date_to}")
        try:
            predictor.preload_caches(journeys)
            print("✅ Cache preloading completed")
        except Exception as e:
            print(f"⚠️ Preload warning: {e}")

        # Run complete prediction for all journeys
        results = {}
        saved_predictions = []
        failed_predictions = []
        
        import time
        start_time = time.time()
        
        for idx, (journey_id, journey_data) in enumerate(journeys.items(), 1):
            journey_start = time.time()
            
            print(f"🔄 Processing journey {idx}/{len(journeys)}: {journey_id}")
            
            prediction = predictor.predict_complete_journey_time(journey_data, journey_id)
            
            # Check if prediction has errors or missing data
            has_error = prediction.get("error") is not None
            has_missing_data = prediction.get("has_missing_data", False)
            
            if has_error:
                failed_predictions.append(journey_id)
                if has_missing_data:
                    print(f"⚠️ Journey {journey_id} excluded due to missing data: {prediction.get('error')}")
                else:
                    print(f"❌ Prediction failed for journey {journey_id}: {prediction.get('error')}")
            else:
                print(f"✅ Prediction completed for journey {journey_id} - {prediction.get('data_coverage_percentage', 0):.1f}% data coverage")
            
            # Only save to database if prediction is successful AND has no missing data
            prediction_id = None
            saved = False
            if not has_error and not has_missing_data:
                try:
                    prediction_id = predictor.save_prediction_normalized(journey_id, journey_data, prediction)
                    if prediction_id:
                        saved_predictions.append(prediction_id)
                        saved = True
                        print(f"💾 Saved {prediction_id} records for journey {journey_id}")
                    else:
                        print(f"❌ Failed to save prediction for journey: {journey_id}")
                except Exception as e:
                    print(f"❌ Exception saving prediction for journey {journey_id}: {str(e)}")
            else:
                if has_missing_data:
                    print(f"⏭️ Skipped saving journey {journey_id} to database (missing data logged to missing_time.txt)")
                else:
                    print(f"⏭️ Skipped saving journey {journey_id} to database (prediction error)")
            
            # Calculate time metrics
            journey_time = time.time() - journey_start
            elapsed_total = time.time() - start_time
            avg_time_per_journey = elapsed_total / idx
            remaining_journeys = len(journeys) - idx
            estimated_remaining = remaining_journeys * avg_time_per_journey
            
            # Progress metrics
            progress_percent = (idx / len(journeys)) * 100
            
            print(f"⏱️ Journey {idx}/{len(journeys)} completed in {journey_time:.2f}s | "
                  f"Progress: {progress_percent:.1f}% | "
                  f"ETA: {estimated_remaining:.0f}s remaining")
            
            results[journey_id] = {
                "journey_data": journey_data,
                "prediction": prediction,
                "route_summary": f"{journey_data['warehouse']} -> {' -> '.join([str(stop['store_code']) for stop in journey_data['stops']])} -> {journey_data['warehouse']}",
                "saved": saved,
                "prediction_id": prediction_id,
                "processing_time": journey_time,
                "has_error": has_error
            }
        
        total_time = time.time() - start_time
        successful_predictions = len(journeys) - len(failed_predictions)
        
        print(f"🏁 Prediction completed!")
        print(f"📈 Total time: {total_time:.2f}s")
        print(f"📊 Success rate: {successful_predictions}/{len(journeys)} ({(successful_predictions/len(journeys)*100):.1f}%)")
        print(f"💾 Saved predictions: {len(saved_predictions)}")
        
        return {
            "results": results,
            "journey_count": len(results),
            "saved_predictions": len(saved_predictions),
            "status": "success",
            "metrics": {
                "total_processing_time": round(total_time, 2),
                "average_time_per_journey": round(total_time / len(journeys), 2),
                "journeys_found": len(journeys),
                "journeys_processed": len(results),
                "success_rate": round((successful_predictions / len(journeys)) * 100, 1),
                "date_range": f"{date_from} to {date_to}"
            },
            "summary": {
                "total_journeys": len(results),
                "successful_predictions": successful_predictions,
                "failed_predictions": len(failed_predictions),
                "predictions_saved": len(saved_predictions)
            }
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}, 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """
    Get saved predictions with optional date filtering using the new normalized structure
    
    Query parameters:
    - date_from: Start date (YYYY-MM-DD)
    - date_to: End date (YYYY-MM-DD)
    - warehouse: Filter by warehouse code
    - journey_id: Filter by specific journey (QYMNUM_0)
    - limit: Number of results (default 100)
    - offset: Pagination offset (default 0)
    """
    try:
        from models import PredictionDetail
        from datetime import datetime
        
        # Get query parameters
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        warehouse = request.args.get('warehouse')
        journey_id = request.args.get('journey_id')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # Build query
        query = PredictionDetail.query
        
        # Apply filters
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
                query = query.filter(PredictionDetail.DAT_0 >= date_from_obj)
            except ValueError:
                return {"error": "Invalid date_from format. Use YYYY-MM-DD"}, 400
        
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
                query = query.filter(PredictionDetail.DAT_0 <= date_to_obj)
            except ValueError:
                return {"error": "Invalid date_to format. Use YYYY-MM-DD"}, 400
        
        if warehouse:
            query = query.filter(PredictionDetail.FCY_0.ilike(f'%{warehouse}%'))
            
        if journey_id:
            query = query.filter(PredictionDetail.QYMNUM_0 == str(journey_id))
        
        # Apply pagination and ordering
        query = query.order_by(PredictionDetail.prediction_created_at.desc())
        total_count = query.count()
        predictions = query.offset(offset).limit(limit).all()
        
        # Convert to JSON
        results = []
        for pred in predictions:
            results.append({
                'id': pred.id,
                'journey_id': pred.QYMNUM_0,
                'prediction_date': pred.DAT_0.isoformat() if pred.DAT_0 else None,
                'warehouse_code': pred.FCY_0,
                'client_code': pred.DLVNAM_0,
                'client_name': pred.DLVNAM_1,
                'pallets': pred.SHPNBR,
                'starting_time': pred.TIM_0,
                'return_time': pred.ZHRD_0,
                'arrival_time': pred.ZHARCL_0,
                'departure_time': pred.ZHDECL_0,
                'is_prediction': pred.is_prediction,
                'data_source': pred.data_source,
                'created_at': pred.prediction_created_at.isoformat() if pred.prediction_created_at else None
            })
        
        return {
            'predictions': results,
            'total_count': total_count,
            'returned_count': len(results),
            'offset': offset,
            'limit': limit,
            'filters': {
                'date_from': date_from,
                'date_to': date_to,
                'warehouse': warehouse,
                'journey_id': journey_id
            },
            'status': 'success'
        }
        
    except Exception as e:
        return {"error": f"Failed to get predictions: {str(e)}"}, 500

@app.route('/predictions/stats', methods=['GET'])
def get_prediction_stats():

    """
    Get statistics about saved predictions using the new normalized structure
    
    Query parameters:
    - date_from: Start date (YYYY-MM-DD) 
    - date_to: End date (YYYY-MM-DD)
    - warehouse: Filter by warehouse code
    """
    try:
        from models import PredictionDetail
        from datetime import datetime
        from sqlalchemy import func
        
        # Get query parameters
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        warehouse = request.args.get('warehouse')
        
        # Build base query
        query = db.session.query(PredictionDetail)
        
        # Apply filters
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
                query = query.filter(PredictionDetail.DAT_0 >= date_from_obj)
            except ValueError:
                return {"error": "Invalid date_from format. Use YYYY-MM-DD"}, 400
        
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
                query = query.filter(PredictionDetail.DAT_0 <= date_to_obj)
            except ValueError:
                return {"error": "Invalid date_to format. Use YYYY-MM-DD"}, 400
        
        if warehouse:
            query = query.filter(PredictionDetail.FCY_0.ilike(f'%{warehouse}%'))
        
        # Get statistics
        total_predictions = query.count()
        
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'stats': {},
                'message': 'No predictions found for the given filters'
            }
        
        # Calculate journey-level statistics
        journey_stats = query.with_entities(
            func.count(func.distinct(PredictionDetail.QYMNUM_0)).label('unique_journeys'),
            func.avg(PredictionDetail.SHPNBR).label('avg_pallets_per_stop'),
            func.sum(PredictionDetail.SHPNBR).label('total_pallets'),
            func.count(PredictionDetail.id).label('total_stops')
        ).first()
        
        # Get warehouse distribution
        warehouse_stats = query.with_entities(
            PredictionDetail.FCY_0,
            func.count(func.distinct(PredictionDetail.QYMNUM_0)).label('journey_count'),
            func.count(PredictionDetail.id).label('stop_count')
        ).group_by(PredictionDetail.FCY_0).all()
        
        # Get date range
        date_range = query.with_entities(
            func.min(PredictionDetail.DAT_0).label('earliest_date'),
            func.max(PredictionDetail.DAT_0).label('latest_date')
        ).first()
        
        return {
            'total_predictions': total_predictions,
            'stats': {
                'unique_journeys': int(journey_stats.unique_journeys) if journey_stats.unique_journeys else 0,
                'total_stops': int(journey_stats.total_stops) if journey_stats.total_stops else 0,
                'avg_pallets_per_stop': round(float(journey_stats.avg_pallets_per_stop), 2) if journey_stats.avg_pallets_per_stop else 0,
                'total_pallets': int(journey_stats.total_pallets) if journey_stats.total_pallets else 0
            },
            'warehouse_distribution': [
                {
                    'warehouse': w.FCY_0, 
                    'journey_count': w.journey_count,
                    'stop_count': w.stop_count
                } 
                for w in warehouse_stats
            ],
            'date_range': {
                'earliest': date_range.earliest_date.isoformat() if date_range.earliest_date else None,
                'latest': date_range.latest_date.isoformat() if date_range.latest_date else None
            },
            'filters': {
                'date_from': date_from,
                'date_to': date_to,
                'warehouse': warehouse
            },
            'status': 'success'
        }
        
    except Exception as e:
        return {"error": f"Failed to get prediction stats: {str(e)}"}, 500

@app.route("/predict_journey", methods=["POST"])
def predict_journey():
    results = {}

    from prediction_engine import JourneyPredictor
    predictor = JourneyPredictor()
    start_time = time.time()
    try:
        data = request.get_json()

        journey_id = data.get("journey_id")
        journey_data = data.get("journey_data")

        if not journey_id or not journey_data:
            return jsonify({"error": "Both journey_id and journey_data must be provided"}), 400

        # Call prediction method
        prediction = predictor.predict_complete_journey_time(journey_data, journey_id)

        journey_time = round(time.time() - start_time, 3)  # processing time in seconds
        has_error = False

        results[journey_id] = {
            "journey_data": journey_data,
            "prediction": prediction,
            "route_summary": f"{journey_data['warehouse']} -> "
                             f"{' -> '.join([str(stop['store_code']) for stop in journey_data['stops']])} -> "
                             f"{journey_data['warehouse']}",
            "processing_time": journey_time,
            "has_error": has_error
        }

        return jsonify(results[journey_id]), 200

    except Exception as e:
        return jsonify({"error": str(e), "has_error": True}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

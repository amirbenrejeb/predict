from flask_sqlalchemy import SQLAlchemy
import os

# Create db instance without app initialization
db = SQLAlchemy()


# Model for stationnement table
class Stationnement(db.Model):
    __tablename__ = 'stationnement'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    DLVNAM0 = db.Column(db.Integer)
    DLVNAM1 = db.Column(db.String(255))
    time = db.Column(db.Integer)
    SHPNBR = db.Column(db.Integer)
    QYMNUM0 = db.Column(db.Integer)
    DATE = db.Column(db.String(255))
    LANMES0 = db.Column(db.String(255))


# Model for client_to_client table
class ClientToClient(db.Model):
    __tablename__ = 'client_to_client'
    NAM_0 = db.Column(db.Text)
    FCY_0 = db.Column(db.Text)
    DEP_0 = db.Column(db.Text)
    DAT_0 = db.Column(db.DateTime)
    QYMNUM_0 = db.Column(db.Text, primary_key=True)
    TIM_0 = db.Column(db.Text)
    ZCAMION_0 = db.Column(db.Text)

    DLVNAM_01 = db.Column(db.Text)
    DLVNAM_11 = db.Column(db.Text)
    DLVNAM_02 = db.Column(db.Text)       # mediumtext maps to Text in SQLAlchemy
    DLVNAM_12 = db.Column(db.Text)       # mediumtext maps to Text in SQLAlchemy

    time_minutes = db.Column(db.BigInteger)


# Example SQLAlchemy model for your data
class DataRow(db.Model):
    __tablename__ = 'to_first_client'
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    FCY0 = db.Column(db.String(100))
    DAT0 = db.Column(db.String(100))
    DLVNAM = db.Column(db.Integer)
    DLVNAM1 = db.Column(db.String(200))
    ZHARCL0 = db.Column(db.Integer)
    ZHCAL = db.Column(db.Integer)
    TIM0 = db.Column(db.Integer)


# Model for from_last_client table
class FromLastClient(db.Model):
    __tablename__ = 'from_last_client'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    DLVNAM0 = db.Column(db.Integer)
    time = db.Column(db.Integer)
    DLVNAM1 = db.Column(db.String(255))
    FCY0 = db.Column(db.String(255))
    date = db.Column(db.String(255))


# Model for prediction results table - New normalized structure
class PredictionDetail(db.Model):
    __tablename__ = 'prediction_details'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    FCY_0 = db.Column(db.String(255))  # Warehouse code
    DAT_0 = db.Column(db.Date)  # Date
    QYMNUM_0 = db.Column(db.String(255))  # Journey code (same for all stops in one journey)
    TIM_0 = db.Column(db.Float)  # Starting time (departure from warehouse)
    ZHRD_0 = db.Column(db.Float)  # Time of return to warehouse (only on last row)
    DLVNAM_0 = db.Column(db.String(255))  # Client code
    SHPNBR = db.Column(db.Integer)  # Number of pallets
    ZHARCL_0 = db.Column(db.Float)  # Time we arrive to the client
    ZHDECL_0 = db.Column(db.Float)  # Time we depart from the client
    DLVNAM_1 = db.Column(db.String(255))  # Client name
    
    # Additional prediction metadata
    is_prediction = db.Column(db.Boolean, default=True)  # Flag to identify prediction vs actual data
    prediction_created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    data_source = db.Column(db.String(255), default='prediction_engine')  # Source of the prediction


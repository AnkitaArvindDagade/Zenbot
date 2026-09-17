const mongoose = require('mongoose');

const connectDB = async () => {
  const uri = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/zenbot';
  try {
    await mongoose.connect(uri);
    console.log(`✅ MongoDB connected: ${mongoose.connection.host}/${mongoose.connection.name}`);
  } catch (err) {
    console.error('❌ MongoDB connection failed:', err.message);
    console.error(
      '   Make sure MongoDB is running locally, or set MONGO_URI in backend/.env to an Atlas connection string.'
    );
    process.exit(1);
  }
};

module.exports = connectDB;

const jwt = require('jsonwebtoken');

module.exports = function generateToken(userId) {
  return jwt.sign({ id: userId }, process.env.JWT_SECRET || 'dev_secret', {
    expiresIn: '30d',
  });
};

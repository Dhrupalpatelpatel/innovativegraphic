const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

module.exports = async (req, res) => {
  try {
    const { name, email, amount, bundle, phone } = req.body;

    // --- Input Validation ---
    if (!name || !email || !amount || !bundle || !phone) {
      return res.status(400).json({ error: 'Missing required fields: name, email, amount, bundle, or phone' });
    }

    const numericAmount = parseFloat(amount);
    if (isNaN(numericAmount) || numericAmount <= 0) {
      return res.status(400).json({ error: 'Invalid order amount provided' });
    }

    // Validate phone (basic Indian phone number check)
    if (!/^[6-9]\d{9}$/.test(phone)) {
      return res.status(400).json({ error: 'Invalid phone number. Must be a 10-digit Indian number starting with 6-9' });
    }

    // Validate email
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return res.status(400).json({ error: 'Invalid email address' });
    }

    // --- Prepare Cashfree Payload ---
    const orderId = `order_${uuidv4()}`; // Unique order ID
    const orderPayload = {
      order_id: orderId,
      order_amount: numericAmount.toFixed(2), // Ensure 2 decimal places (e.g., 99.00)
      order_currency: 'INR',
      customer_details: {
        customer_id: email, // Unique identifier
        customer_name: name,
        customer_email: email,
        customer_phone: phone,
      },
      order_meta: {
        return_url: `https://innovativegraphic.vercel.app/thank-you?order_id={order_id}`,
        notify_url: 'https://innovativegraphic.vercel.app/api/webhook', // Enable webhook
      },
      order_note: `Order for bundle: ${bundle}`,
    };

    // --- Cashfree API Call ---
    const cashfreeHeaders = {
      'Content-Type': 'application/json',
      'x-client-id': process.env.CASHFREE_APP_ID,
      'x-client-secret': process.env.CASHFREE_SECRET_KEY,
      'x-api-version': '2023-08-01', // Latest API version
      'x-request-id': `order-${orderId}`,
    };

    const cashfreeApiUrl = process.env.CASHFREE_ENV === 'production'
      ? 'https://api.cashfree.com/pg/orders'
      : 'https://sandbox.cashfree.com/pg/orders';

    const response = await axios.post(cashfreeApiUrl, orderPayload, { headers: cashfreeHeaders });

    if (!response.data || !response.data.payment_session_id) {
      throw new Error('Failed to retrieve payment session ID from Cashfree');
    }

    // --- Store Order in Database (Assumed) ---
    // Example: Save to a database like MongoDB or PostgreSQL
    /*
    await Order.create({
      orderId,
      userEmail: email,
      userName: name,
      userPhone: phone,
      bundle,
      amount: numericAmount,
      status: 'pending',
      createdAt: new Date(),
    });
    */

    // --- Return Response ---
    return res.status(200).json({
      paymentSessionId: response.data.payment_session_id,
      orderId,
    });

  } catch (error) {
    console.error('Error creating Cashfree order:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
    });

    if (error.response) {
      const { status, data } = error.response;
      return res.status(status || 500).json({
        error: data?.message || 'Payment gateway error',
        details: {
          type: data?.type,
          code: data?.code,
        },
      });
    } else if (error.request) {
      return res.status(504).json({ error: 'Payment gateway timeout. Please try again' });
    } else {
      return res.status(500).json({ error: 'Internal server error' });
    }
  }
};
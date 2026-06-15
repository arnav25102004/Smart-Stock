document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Render Stock Levels Panel ---
    const stockData = [
        { name: 'Rice', amount: 15.5 },
        { name: 'Sugar', amount: 8.2 },
        { name: 'Dal', amount: 4.1 },
        { name: 'Tea', amount: 2.0 },
        { name: 'Oil', amount: 5.3 },
        { name: 'Flour', amount: 3.8 },
    ];

    const stockGrid = document.getElementById('stock-grid');
    
    stockData.forEach(item => {
        let colorClass = 'text-success'; // > 5
        let borderColor = 'border-gray-700';
        let badge = '';

        if (item.amount < 2) {
            colorClass = 'text-danger';
            borderColor = 'border-danger/30';
            badge = `<span class="absolute top-2 right-2 bg-danger text-white text-[10px] font-bold px-2 py-0.5 rounded animate-pulse">LOW STOCK</span>`;
        } else if (item.amount <= 5) {
            colorClass = 'text-warning';
        }

        const card = document.createElement('div');
        card.className = `relative bg-cardBg border ${borderColor} rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow`;
        card.innerHTML = `
            ${badge}
            <div class="text-sm text-gray-400 font-medium mb-1">${item.name}</div>
            <div class="text-2xl font-bold ${colorClass}">${item.amount.toFixed(1)} <span class="text-sm font-normal text-gray-500">kg/L</span></div>
        `;
        stockGrid.appendChild(card);
    });

    // --- 2. Render Demand Forecast Panel ---
    const forecastData = [
        { category: 'GROCERY', qty: 142, trend: 'up' },
        { category: 'DAIRY', qty: 85, trend: 'up' },
        { category: 'BEVERAGES', qty: 110, trend: 'down' },
        { category: 'CLEANING', qty: 34, trend: 'up' }
    ];

    const forecastGrid = document.getElementById('forecast-grid');
    
    forecastData.forEach(item => {
        const trendIcon = item.trend === 'up' 
            ? `<svg class="w-4 h-4 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>`
            : `<svg class="w-4 h-4 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path></svg>`;

        const card = document.createElement('div');
        card.className = "bg-cardBg border border-gray-700 rounded-xl p-4 flex flex-col justify-between shadow-sm";
        card.innerHTML = `
            <div class="text-xs text-gray-400 font-medium mb-2">${item.category}</div>
            <div class="flex items-end items-center justify-between">
                <div class="text-xl font-bold text-white">${item.qty} <span class="text-xs font-normal text-gray-500">units</span></div>
                ${trendIcon}
            </div>
        `;
        forecastGrid.appendChild(card);
    });

    // --- 3. Chat / Query Logic ---
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const chatWindow = document.getElementById('chat-window');
    const suggestionChips = document.querySelectorAll('.chip');

    // Handle suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.textContent;
            queryInput.focus();
        });
    });

    const addMessage = (text, isUser, data = null) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `p-3 text-sm max-w-[85%] border shadow-sm bubble-enter ${
            isUser 
            ? 'bg-accent/10 border-accent/30 text-white rounded-lg rounded-br-none self-end' 
            : 'bg-gray-800 border-gray-700 text-gray-100 rounded-lg rounded-tl-none self-start relative'
        }`;
        
        let innerHTML = `<p>${text}</p>`;

        // If it's a bot response with data, add metadata pills
        if (!isUser && data) {
            innerHTML += `
                <div class="flex gap-2 mt-2 pt-2 border-t border-gray-600/50">
                    <span class="text-[10px] bg-gray-700 px-2 py-0.5 rounded text-gray-300">Lang: <span class="text-white">${data.language}</span></span>
                    <span class="text-[10px] bg-gray-700 px-2 py-0.5 rounded text-gray-300">Intent: <span class="text-accent">${data.intent}</span></span>
                </div>
            `;
        }

        msgDiv.innerHTML = innerHTML;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const addTypingIndicator = () => {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'bg-gray-800 border border-gray-700 p-3 rounded-lg rounded-tl-none self-start max-w-[85%] shadow-sm bubble-enter flex space-x-1 items-center h-[40px]';
        msgDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return id;
    };

    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text) return;

        // Add user message
        addMessage(text, true);
        queryInput.value = '';

        // Add loading indicator
        const typingId = addTypingIndicator();

        try {
            const response = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            
            // Remove typing indicator
            document.getElementById(typingId).remove();

            if (response.ok) {
                addMessage(data.response, false, data);
            } else {
                addMessage(data.error || 'Something went wrong.', false);
            }

        } catch (err) {
            // Remove typing indicator
            document.getElementById(typingId).remove();
            addMessage('Error connecting to backend API (Is it running on port 5000?)', false);
            console.error('Fetch error:', err);
        }
    });

});
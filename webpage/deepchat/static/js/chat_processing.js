$(document).ready(function() {

    // Extract the user input from the field.
    let chatForm = $('.chat-form');
    let userMessage = chatForm.find('input[name="message"]');
    let submitButton = chatForm.find('.chat-form-submit')

    submitButton.on('click', function(e) {

        // Not sure why the specificity is required here (it isn't for my machine)
        // but the deployed app always selects the first chat box otherwise.
        let dataName = $('.chat-box.active .chat-form-submit').data('name');
        console.log('Event triggered for button with data name:', dataName);
        userMessage = $('#'+dataName);

        if (!userMessage.val()) {  // don't do anything on empty input
            return;
        }

        let chatLog = $('#'+dataName+'-chat-log');
        let messageRow = $('<div/>').attr('class', 'row message');

        messageRow.append($('<div>' + userMessage.val() + '</div>')
                .addClass('user-message text-left')
                .addClass('col-md-8 col-md-push-2')
                .addClass('col-sm-10 col-sm-push-2'));
        messageRow.append($('<div>User</div>')
                .addClass('user-name text-left')
                .addClass('col-md-2 col-md-offset-2'));
        messageRow.append($('<hr/>'));

        chatLog.append(messageRow);

        // Submit a POST request to collect user input.
        $.post('/chat/' + dataName + '/', {
            "user_message": userMessage.val()
        }, function(data) {
            userMessage.val("");
            console.log('Response received from bot', data.bot_name)
            $('#'+dataName+'-chat-log').append($('<div/>').addClass('row message')
                    .append($('<div/>')
                            .addClass('bot-name text-left col-md-2 col-sm-2')
                            .text('Botty'))
                    .append($('<div/>')
                            .addClass('bot-message text-left col-md-8 col-sm-9')
                            .text(data.response))
                    .append($('<hr/>')));
            chatLog.scrollTop(chatLog.first().scrollHeight);
        });
    });

    userMessage.on('keyup', function(e) {
        if (e.which == 13) {
            $(this).parent().find('.btn').click();
        }
    })

});

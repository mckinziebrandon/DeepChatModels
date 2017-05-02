$(document).ready(function() {

    // Extract the user input from the field.
    let chatForm = $('.chat-form');
    let userMessage = chatForm.find('input[name="message"]');
    let submitButton = chatForm.find('.chat-form-submit')

    submitButton.on('click', function(e) {

        let dataName = $(this).data('name');
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
        $.post('/', {
            "message": userMessage.val(),
            "dataName": dataName
        }, function(data) {
            console.log('Response data:', data);
            $('#'+dataName+'-chat-log').append($('<div/>').addClass('row message')
                    .append($('<div/>')
                            .addClass('bot-name text-left col-md-2 col-sm-2')
                            .text('Botty'))
                    .append($('<div/>')
                            .addClass('bot-message text-left col-md-8 col-sm-9')
                            .text(data))
                    .append($('<hr/>')));
            chatLog.scrollTop(chatLog.first().scrollHeight);
            userMessage.val("");
        });
    });

    userMessage.on('keyup', function(e) {
        if (e.which == 13) {
            $(this).parent().find('.btn').click();
        }
    })

});


$('#frm1').submit(function(e){
    e.preventDefault();
    $.post('chat/', $(this).serialize(), function(data){
        $('#chat-log').append("generic response");
    });
});